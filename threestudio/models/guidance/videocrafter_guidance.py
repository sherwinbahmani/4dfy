from dataclasses import dataclass, field
import inspect
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import C, cleanup, parse_version
from threestudio.utils.typing import *

from omegaconf import OmegaConf
from einops import rearrange

import numpy as np

@threestudio.register("videocrafter-guidance")
class VideoCrafterGuidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        config: str = None
        pretrained_model_name_or_path: Optional[str] = None
        enable_memory_efficient_attention: bool = False
        enable_sequential_cpu_offload: bool = False
        enable_attention_slicing: bool = False
        enable_channels_last_format: bool = False
        guidance_scale: float = 100.0
        grad_clip: Optional[
            Any
        ] = None 
        half_precision_weights: bool = True

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98
        max_step_percent_annealed: float = 0.5
        anneal_start_step: Optional[int] = None

        use_sjc: bool = False
        var_red: bool = True
        weighting_strategy: str = "sds"

        token_merging: bool = False
        token_merging_params: Optional[dict] = field(default_factory=dict)

        view_dependent_prompting: bool = True
        low_ram_vae: int = -1
        use_hifa: bool = False
        width_vid: int = 128
        height_vid: int = 80

        motion_amp_scale: float = 1.0

        fps: int = 28

    cfg: Config

    def configure(self) -> None:
        import sys, os
        sys.path.insert(1, os.path.join(sys.path[0], "threestudio", 'models', 'guidance', 'videocrafter'))
        from threestudio.models.guidance.videocrafter.utils.utils import instantiate_from_config
        from threestudio.models.guidance.videocrafter.scripts.evaluation.funcs import load_model_checkpoint
        from threestudio.models.guidance.videocrafter.lvdm.models.samplers.ddim import DDIMSampler

        threestudio.info(f"Loading VideoCrafter ...")
        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )

        config = OmegaConf.load(self.cfg.config)
        self.model_config = config.pop("model", OmegaConf.create())
        self.model = instantiate_from_config(self.model_config)
        self.model = self.model.to(self.device)
        self.model = load_model_checkpoint(self.model, self.cfg.pretrained_model_name_or_path)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        cleanup()
        self.scheduler = DDIMSampler(self.model)
        self.num_train_timesteps = self.model_config.params.timesteps
        min_step_percent = C(self.cfg.min_step_percent, 0, 0)
        max_step_percent = C(self.cfg.max_step_percent, 0, 0)
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)

        self.alphas = self.model.alphas_cumprod

        self.grad_clip_val: Optional[float] = None

        threestudio.info(f"Loaded Video Crafter!")

    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
        self,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."],
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        return self.unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
        ).sample.to(input_dtype)
    
    def encode_first_stage(self, x):
        if self.model.encoder_type == "2d" and x.dim() == 5:
            b, _, t, _, _ = x.shape
            x = rearrange(x, 'b c t h w -> (b t) c h w')
            reshape_back = True
        else:
            reshape_back = False

        if self.cfg.low_ram_vae > 0:
            vnum = self.cfg.low_ram_vae
            mask_vae = torch.randperm(x.shape[0]) < vnum
            if not mask_vae.all():
                with torch.no_grad():
                    posterior_mask = torch.cat(
                        [
                            self.model.first_stage_model.encode(
                                x[~mask_vae][i : i + 1].to(self.weights_dtype)
                            ).sample(None)
                            for i in range(x.shape[0] - vnum)
                        ],
                        dim=0,
                    )
            posterior = torch.cat(
                [
                    self.model.first_stage_model.encode(
                        x[mask_vae][i : i + 1].to(self.weights_dtype)
                    ).sample(None)
                    for i in range(vnum)
                ],
                dim=0,
            )
            encoder_posterior = torch.zeros(
                x.shape[0],
                *posterior.shape[1:],
                device=posterior.device,
                dtype=posterior.dtype,
            )
            if not mask_vae.all():
                encoder_posterior[~mask_vae] = posterior_mask
            encoder_posterior[mask_vae] = posterior
        else:
            encoder_posterior = self.model.first_stage_model.encode(x.to(self.weights_dtype)).sample(None)
        results = encoder_posterior*self.model.scale_factor
        if reshape_back:
            results = rearrange(results, '(b t) c h w -> b c t h w', b=b,t=t)
        return results
    
    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
        self, imgs: Float[Tensor, "B 3 N 320 512"], normalize: bool = True
    ) -> Float[Tensor, "B 4 40 64"]:
        if len(imgs.shape) == 4:
            print("Only given an image an not video")
            imgs = imgs[:, :, None]
        batch_size, channels, num_frames, height, width = imgs.shape
        imgs = imgs.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, height, width)
        input_dtype = imgs.dtype
        if normalize:
            imgs = imgs * 2.0 - 1.0
        latents = self.encode_first_stage(imgs)

        latents = (
            latents[None, :]
            .reshape(
                (
                    batch_size,
                    num_frames,
                    -1,
                )
                + latents.shape[2:]
            )
            .permute(0, 2, 1, 3, 4)
        )
        return latents.to(input_dtype)
    
    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(self, latents):
        raise NotImplementedError

    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
        # Move the self.alphas_cumprod to device to avoid redundant CPU to GPU data movement
        # for the subsequent add_noise calls
        alphas_cumprod = self.alphas
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def compute_grad_sds(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        text_embeddings: Float[Tensor, "BB 77 768"],
        t: Int[Tensor, "B"],
        pred_rgb_512,
    ):
        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)  # TODO: use torch generator
            latents_noisy = self.add_noise(latents, noise, t)
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            cond = {'c_crossattn': [text_embeddings], 'fps': torch.full((2,), self.cfg.fps, device=latents.device)}
            noise_pred = self.model.apply_model(
                latent_model_input,
                torch.cat([t] * 2),
                cond,
                x0=None,
                temporal_length=16
                )
            
        # perform guidance (high scale from paper!)
        noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred = noise_pred_text + self.cfg.guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        if self.cfg.weighting_strategy == "sds":
            # w(t), sigma_t^2
            w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        elif self.cfg.weighting_strategy == "uniform":
            w = 1
        elif self.cfg.weighting_strategy == "fantasia3d":
            w = (self.alphas[t] ** 0.5 * (1 - self.alphas[t])).view(-1, 1, 1, 1)
        else:
            raise ValueError(
                f"Unknown weighting strategy: {self.cfg.weighting_strategy}"
            )
        
        if self.cfg.use_hifa:
            latents_denoised = (latents_noisy - (1 - self.alphas[t]) ** 0.5 * noise_pred) / self.alphas[t] ** 0.5
            error = latents - latents_denoised.detach()
            latent_loss = torch.sum(error ** 2) / 2 * w * self.size_scale

            rgb_denoised = self.decode_latents(latents_denoised.to(self.weights_dtype))
            color_coeff = torch.Tensor([1, 1, 1]).reshape(1, 3, 1, 1, 1).to(self.device)
            error_rgb = pred_rgb_512 * color_coeff - rgb_denoised * color_coeff
            rgb_loss = torch.sum(error_rgb ** 2) / 2 * w * self.size_scale
            grad = latent_loss + 0.1 * rgb_loss
        else:
            score = noise_pred - noise
            if self.cfg.motion_amp_scale != 1.0:
                score_mean = score.mean(2, keepdim=True)
                score = score_mean + self.cfg.motion_amp_scale*(score - score_mean)
            grad = w * score
        return grad

    def __call__(
        self,
        rgb: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        rgb_as_latents: bool = False,
        num_frames: int = 16,
        **kwargs,
    ):
        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        batch_size = rgb_BCHW.shape[0] // num_frames
        latents: Float[Tensor, "B 4 40 72"]
        if kwargs['train_dynamic_camera']:
            elevation = elevation[[0]]
            azimuth = azimuth[[0]]
            camera_distances = camera_distances[[0]]
        if rgb_as_latents:
            latents = F.interpolate(
                rgb_BCHW, (40, 64), mode="bilinear", align_corners=False
            )
        else:
            rgb_BCHW_512 = F.interpolate(
                rgb_BCHW, (320, 512), mode="bilinear", align_corners=False
            )
            rgb_BCHW_512 = rgb_BCHW_512.permute(1, 0, 2, 3)[None]
            latents = self.encode_images(rgb_BCHW_512)
        text_embeddings = prompt_utils.get_text_embeddings(
            elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
        )
        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(
            self.min_step,
            self.max_step + 1,
            [batch_size],
            dtype=torch.long,
            device=self.device,
        )
        grad = self.compute_grad_sds(latents, text_embeddings, t, rgb_BCHW_512)
        grad = torch.nan_to_num(grad)
        # clip grad for stable training?
        if self.grad_clip_val is not None:
            grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)

        # loss = SpecifyGradient.apply(latents, grad)
        # SpecifyGradient is not straghtforward, use a reparameterization trick instead
        target = (latents - grad).detach()
        # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
        loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size
        return {
            "loss_sds_video": loss_sds,
            "grad_norm": grad.norm(),
        }

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # clip grad for stable training as demonstrated in
        # Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation
        # http://arxiv.org/abs/2303.15413
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)

        min_step_percent = C(self.cfg.min_step_percent, epoch, global_step)
        max_step_percent = C(self.cfg.max_step_percent, epoch, global_step)
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)