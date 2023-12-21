from dataclasses import dataclass, field
import inspect
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMScheduler, StableDiffusionPipeline
from diffusers.utils.import_utils import is_xformers_available

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import C, cleanup, parse_version
from threestudio.utils.typing import *

from transformers import CLIPTokenizer, CLIPTextModel

import numpy as np

@threestudio.register("zeroscope-guidance")
class ZeroscopeGuidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        pretrained_model_name_or_path: str = None
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

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading Stable Diffusion ...")

        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )

        pipe_kwargs = {
            "tokenizer": None,
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
            "torch_dtype": self.weights_dtype,
        }
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            **pipe_kwargs,
        ).to(self.device)

        # Extra modules
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.cfg.pretrained_model_name_or_path, subfolder="tokenizer",
            torch_dtype=self.weights_dtype
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.cfg.pretrained_model_name_or_path, subfolder="text_encoder",
            torch_dtype=self.weights_dtype
        )
        self.text_encoder = self.text_encoder.to(self.device)


        if self.cfg.enable_memory_efficient_attention:
            if parse_version(torch.__version__) >= parse_version("2"):
                threestudio.info(
                    "PyTorch2.0 uses memory efficient attention by default."
                )
            elif not is_xformers_available():
                threestudio.warn(
                    "xformers is not available, memory efficient attention is not enabled."
                )
            else:
                self.pipe.enable_xformers_memory_efficient_attention()

        if self.cfg.enable_sequential_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()

        if self.cfg.enable_attention_slicing:
            self.pipe.enable_attention_slicing(1)

        if self.cfg.enable_channels_last_format:
            self.pipe.unet.to(memory_format=torch.channels_last)

        # del self.pipe.text_encoder
        cleanup()

        # Create model
        self.vae = self.pipe.vae.eval()
        self.unet = self.pipe.unet.eval()

        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)

        if self.cfg.token_merging:
            import tomesd

            tomesd.apply_patch(self.unet, **self.cfg.token_merging_params)

        if self.cfg.use_sjc:
            # score jacobian chaining use DDPM
            self.scheduler = DDPMScheduler.from_pretrained(
                self.cfg.pretrained_model_name_or_path,
                subfolder="scheduler",
                torch_dtype=self.weights_dtype,
                beta_start=0.00085,
                beta_end=0.0120,
                beta_schedule="scaled_linear",
            )
        else:
            self.scheduler = DDIMScheduler.from_pretrained(
                self.cfg.pretrained_model_name_or_path,
                subfolder="scheduler",
                torch_dtype=self.weights_dtype,
            )

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * self.cfg.min_step_percent)
        self.max_step = int(self.num_train_timesteps * self.cfg.max_step_percent)

        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(
            self.device
        )
        if self.cfg.use_sjc:
            # score jacobian chaining need mu
            self.us: Float[Tensor, "..."] = torch.sqrt((1 - self.alphas) / self.alphas)

        self.grad_clip_val: Optional[float] = None

        # Extra for latents
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        threestudio.info(f"Loaded Stable Diffusion!")

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
    
    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
        self, imgs: Float[Tensor, "B 3 N 320 576"], normalize: bool = True
    ) -> Float[Tensor, "B 4 40 72"]:
        # breakpoint()
        if len(imgs.shape) == 4:
            print("Only given an image an not video")
            imgs = imgs[:, :, None]
        # breakpoint()
        batch_size, channels, num_frames, height, width = imgs.shape
        imgs = imgs.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, height, width)
        input_dtype = imgs.dtype
        if normalize:
            imgs = imgs * 2.0 - 1.0
        # breakpoint()
        if self.cfg.low_ram_vae > 0:
            vnum = self.cfg.low_ram_vae
            mask_vae = torch.randperm(imgs.shape[0]) < vnum
            with torch.no_grad():
                posterior_mask = torch.cat(
                    [
                        self.vae.encode(
                            imgs[~mask_vae][i : i + 1].to(self.weights_dtype)
                        ).latent_dist.sample()
                        for i in range(imgs.shape[0] - vnum)
                    ],
                    dim=0,
                )
            posterior = torch.cat(
                [
                    self.vae.encode(
                        imgs[mask_vae][i : i + 1].to(self.weights_dtype)
                    ).latent_dist.sample()
                    for i in range(vnum)
                ],
                dim=0,
            )
            posterior_full = torch.zeros(
                imgs.shape[0],
                *posterior.shape[1:],
                device=posterior.device,
                dtype=posterior.dtype,
            )
            posterior_full[~mask_vae] = posterior_mask
            posterior_full[mask_vae] = posterior
            latents = posterior_full * self.vae.config.scaling_factor
        else:
            posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
            latents = posterior.sample() * self.vae.config.scaling_factor

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
        # TODO: Make decoding align with previous version
        latents = 1 / self.vae.config.scaling_factor * latents

        batch_size, channels, num_frames, height, width = latents.shape
        latents = latents.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, height, width)

        image = self.vae.decode(latents).sample
        video = (
            image[None, :]
            .reshape(
                (
                    batch_size,
                    num_frames,
                    -1,
                )
                + image.shape[2:]
            )
            .permute(0, 2, 1, 3, 4)
        )
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        video = video.float()
        return video

    def compute_grad_sds(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        text_embeddings: Float[Tensor, "BB 77 768"],
        t: Int[Tensor, "B"],
    ):
        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)  # TODO: use torch generator
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            noise_pred = self.forward_unet(
                latent_model_input,
                torch.cat([t] * 2),
                encoder_hidden_states=text_embeddings,
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

        grad = w * (noise_pred - noise)
        return grad

    def compute_grad_sjc(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        text_embeddings: Float[Tensor, "BB 77 768"],
        t: Int[Tensor, "B"],
    ):
        sigma = self.us[t]
        sigma = sigma.view(-1, 1, 1, 1)
        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)  # TODO: use torch generator
            y = latents

            zs = y + sigma * noise
            scaled_zs = zs / torch.sqrt(1 + sigma**2)

            # pred noise
            latent_model_input = torch.cat([scaled_zs] * 2, dim=0)
            noise_pred = self.forward_unet(
                latent_model_input,
                torch.cat([t] * 2),
                encoder_hidden_states=text_embeddings,
            )

            # perform guidance (high scale from paper!)
            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_text + self.cfg.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            Ds = zs - sigma * noise_pred

            if self.cfg.var_red:
                grad = -(Ds - y) / sigma
            else:
                grad = -(Ds - zs) / sigma

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
                rgb_BCHW, (40, 72), mode="bilinear", align_corners=False
            )
        else:
            rgb_BCHW_512 = F.interpolate(
                rgb_BCHW, (320, 576), mode="bilinear", align_corners=False
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
        if self.cfg.use_sjc:
            grad = self.compute_grad_sjc(latents, text_embeddings, t)
        else:
            grad = self.compute_grad_sds(latents, text_embeddings, t)
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

        # t annealing from ProlificDreamer
        if (
            self.cfg.anneal_start_step is not None
            and global_step > self.cfg.anneal_start_step
        ):
            self.max_step = int(
                self.num_train_timesteps * self.cfg.max_step_percent_annealed
            )