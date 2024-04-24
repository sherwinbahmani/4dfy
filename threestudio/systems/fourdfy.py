import os
from dataclasses import dataclass

import torch
import numpy as np

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.ops import binary_cross_entropy, dot, TVLoss
from threestudio.utils.typing import *


@threestudio.register("fourdfy-system")
class Fourdfy(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        stage: str = "coarse"
        visualize_samples: bool = False
        prob_multi_view: Optional[float] = None
        prob_single_view_video: Optional[float] = None
        eval_depth_range_perc: Tuple[float, float] = (10, 99) # Adjust manually based on object, near far depth bounds percentage in (0, 100)
        freeze_static_modules: Optional[bool] = False

    cfg: Config

    def configure(self) -> None:
        # set up geometry, material, background, renderer
        super().configure()
        self.simultan = self.cfg.get("simultan", False)
        self.static = self.cfg.geometry.pos_encoding_config.get("static", True)
        self.prob_multi_view = self.cfg.get("prob_multi_view", None)
        self.single_view_img = self.cfg.prob_single_view_video not in [1.0, None] or self.static
        self.guidance = None
        self.prompt_processor = None
        self.prompt_utils = None
        self.geometry_encoding = self.geometry.encoding.encoding
        self.tv_loss = TVLoss() if self.C(self.cfg.loss.lambda_deformation) > 0 else None
        if self.prob_multi_view not in [0.0, None]:
            self.cfg.prompt_processor_multi_view["prompt"] = self.cfg.prompt_processor["prompt"]
            self.guidance_multi_view = threestudio.find(self.cfg.guidance_type_multi_view)(self.cfg.guidance_multi_view)
            self.prompt_processor_multi_view = threestudio.find(self.cfg.prompt_processor_type_multi_view)(
                self.cfg.prompt_processor_multi_view
            )
            self.prompt_utils_multi_view = self.prompt_processor_multi_view()

            self.guidance = self.guidance_multi_view
            self.prompt_processor = self.prompt_processor_multi_view
            self.prompt_utils = self.prompt_utils_multi_view
        if not self.static:
            self.cfg.prompt_processor_video["prompt"] = self.cfg.prompt_processor["prompt"]
            self.guidance_video = threestudio.find(self.cfg.guidance_type_video)(self.cfg.guidance_video)
            self.prompt_processor_video = threestudio.find(self.cfg.prompt_processor_type_video)(
                self.cfg.prompt_processor_video
            )
            self.prompt_utils_video = self.prompt_processor_video()
            if self.guidance is None:
                self.guidance = self.guidance_video
                self.prompt_processor = self.prompt_processor_video
                self.prompt_utils = self.prompt_utils_video
        if self.single_view_img:
            self.guidance_single_view = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
            self.prompt_processor_single = threestudio.find(self.cfg.prompt_processor_type)(
                self.cfg.prompt_processor
            )
            self.prompt_utils_single_view = self.prompt_processor_single()
            if self.guidance is None:
                self.guidance = self.guidance_single_view
                self.prompt_processor = self.prompt_processor_single
                self.prompt_utils = self.prompt_utils_single_view

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        if self.cfg.stage == "geometry":
            out = self.renderer(**batch, render_normal=True, render_rgb=False)
        else:
            if not self.static:
                render_outs = []
                # TODO: Handle batch size higher than 1
                batch["frame_times"] = batch["frame_times"].flatten()
                for frame_idx, frame_time in enumerate(batch["frame_times"].tolist()):
                    self.geometry_encoding.frame_time = frame_time
                    if batch['train_dynamic_camera']:
                        batch_frame = {}
                        for k_frame, v_frame in batch.items():
                            if isinstance(v_frame, torch.Tensor):
                                if v_frame.shape[0] == batch["frame_times"].shape[0]:
                                    v_frame_up = v_frame[[frame_idx]].clone()
                                else:
                                    v_frame_up = v_frame.clone()
                            else:
                                v_frame_up = v_frame
                            batch_frame[k_frame] = v_frame_up
                        render_out = self.renderer(**batch_frame)
                    else:
                        render_out = self.renderer(**batch)
                    render_outs.append(render_out)
                out = {}
                for k, v in render_out.items():
                    if v is not None:
                        out[k] = torch.cat([render_out_i[k] for render_out_i in render_outs])
                    else:
                        out[k] = v
            else:
                out = self.renderer(**batch)
        return out

    def on_fit_start(self) -> None:
        super().on_fit_start()
        if self.cfg.freeze_static_modules:
            for p in self.parameters():
                p.requires_grad = False

    def training_step(self, batch, batch_idx):
        is_video = batch["is_video"]
        batch_size = batch['c2w'].shape[0]
        if batch['train_dynamic_camera']:
            batch_size = batch_size // batch['frame_times'].shape[0]
        if is_video: 
            guidance = self.guidance_video
            prompt_utils = self.prompt_utils_video
            static = self.static
            self.geometry_encoding.is_video = True
            self.geometry_encoding.set_temp_param_grad(True)
        else:
            if batch['single_view']:
                guidance = self.guidance_single_view
                prompt_utils = self.prompt_utils_single_view
            else:
                guidance = self.guidance_multi_view
                prompt_utils = self.prompt_utils_multi_view
            static = True
            num_static_frames = 1 # Use a single random time for static guidance
            batch["frame_times"] = batch["frame_times"][torch.randperm(batch["frame_times"].shape[0])][:num_static_frames]
            self.geometry_encoding.is_video = False
            if self.cfg.freeze_static_modules:
                self.geometry_encoding.set_temp_param_grad(True)
            else:
                self.geometry_encoding.set_temp_param_grad(False)
        out = self(batch)
        if not self.static:
            if static:
                batch['num_frames'] = num_static_frames
            else:
                batch['num_frames'] = self.cfg.geometry.pos_encoding_config.num_frames 

        if self.cfg.stage == "geometry":
            guidance_inp = out["comp_normal"]
            guidance_out = guidance(
                guidance_inp, self.prompt_utils, **batch, rgb_as_latents=False
            )
        else:
            guidance_inp = out["comp_rgb"]
            if static:
                guidance_out_list = [guidance(guidance_inp_i, prompt_utils, **batch, rgb_as_latents=False) for guidance_inp_i in guidance_inp.split(batch_size)]
                guidance_out = {k: torch.zeros_like(v) for k, v in guidance_out_list[0].items()}
                for guidance_out_i in guidance_out_list:
                    for k, v in guidance_out.items():
                        guidance_out[k] = v + guidance_out_i[k]
                for k, v in guidance_out.items():
                    guidance_out[k] = v / len(guidance_out_list)
            else:
                guidance_out = guidance(
                    guidance_inp, prompt_utils, **batch, rgb_as_latents=False
                )
        loss = 0.0

        for name, value in guidance_out.items():
            self.log(f"train/{name}", value)
            if name.startswith("loss_"):
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])

        if self.cfg.stage == "coarse":
            if self.C(self.cfg.loss.lambda_orient) > 0:
                if "normal" not in out:
                    raise ValueError(
                        "Normal is required for orientation loss, no normal is found in the output."
                    )
                loss_orient = (
                    out["weights"].detach()
                    * dot(out["normal"], out["t_dirs"]).clamp_min(0.0) ** 2
                ).sum() / (out["opacity"] > 0).sum()
                self.log("train/loss_orient", loss_orient)
                loss += loss_orient * self.C(self.cfg.loss.lambda_orient)

            if self.C(self.cfg.loss.lambda_sparsity) > 0:
                loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
                self.log("train/loss_sparsity", loss_sparsity)
                loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

            if self.C(self.cfg.loss.lambda_opaque) > 0:
                opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
                loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
                self.log("train/loss_opaque", loss_opaque)
                loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)

            # z variance loss proposed in HiFA: http://arxiv.org/abs/2305.18766
            # helps reduce floaters and produce solid geometry
            if self.C(self.cfg.loss.lambda_z_variance) > 0:
                loss_z_variance = out["z_variance"][out["opacity"] > 0.5].mean()
                self.log("train/loss_z_variance", loss_z_variance)
                loss += loss_z_variance * self.C(self.cfg.loss.lambda_z_variance)
            if self.C(self.cfg.loss.lambda_deformation) > 0 and "deformation" in out and out["deformation"] is not None:
                loss_deformation = self.tv_loss(out["deformation"].permute(3, 0, 1, 2))
                self.log("train/loss_deformation", loss_deformation)
                loss += loss_deformation * self.C(self.cfg.loss.lambda_deformation)

        elif self.cfg.stage == "geometry":
            loss_normal_consistency = out["mesh"].normal_consistency()
            self.log("train/loss_normal_consistency", loss_normal_consistency)
            loss += loss_normal_consistency * self.C(
                self.cfg.loss.lambda_normal_consistency
            )

            if self.C(self.cfg.loss.lambda_laplacian_smoothness) > 0:
                loss_laplacian_smoothness = out["mesh"].laplacian()
                self.log("train/loss_laplacian_smoothness", loss_laplacian_smoothness)
                loss += loss_laplacian_smoothness * self.C(
                    self.cfg.loss.lambda_laplacian_smoothness
                )
        elif self.cfg.stage == "texture":
            pass
        else:
            raise ValueError(f"Unknown stage {self.cfg.stage}")

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],
            name="validation_step",
            step=self.true_global_step,
        )

        if not self.static:
            batch_video = {k: v for k, v in batch.items() if k != "frame_times"}
            batch_video["frame_times"] = batch["frame_times_video"]
            out_video = self(batch_video)
            self.save_image_grid(
            f"it{self.true_global_step}-{batch['index'][0]}_video.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": out_video["comp_rgb"],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out_video
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out_video["comp_normal"],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out_video
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out_video["opacity"],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],
            name="validation_step",
            step=self.true_global_step,
            video=True
        )

        if self.cfg.visualize_samples:
            self.save_image_grid(
                f"it{self.true_global_step}-{batch['index'][0]}-sample.png",
                [
                    {
                        "type": "rgb",
                        "img": self.guidance_single_view.sample(
                            self.prompt_utils_single_view, **batch, seed=self.global_step
                        )[0],
                        "kwargs": {"data_format": "HWC"},
                    },
                    {
                        "type": "rgb",
                        "img": self.guidance_single_view.sample_lora(self.prompt_utils_single_view, **batch)[0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ],
                name="validation_step_samples",
                step=self.true_global_step,
            )

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.out_depths = []
        out = self(batch)
        depth = out["depth"][0, :, :, 0].detach().cpu().numpy()
        self.out_depths.append(depth)
        if "comp_rgb" in out:
            self.save_image_grid(
                f"it{self.true_global_step}-test/{batch['index'][0]}.png",
                (
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_rgb"][0],
                            "kwargs": {"data_format": "HWC"},
                        },
                    ]
                ),
                name="test_step",
                step=self.true_global_step,
            )
        if "comp_normal" in out:
            self.save_image_grid(
                f"it{self.true_global_step}-test-normal/{batch['index'][0]}.png",
                (
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_normal"][0],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                ),
                name="test_step",
                step=self.true_global_step,
            )
        if not self.static:
            batch_static = {k: v for k, v in batch.items() if k != "frame_times"}
            batch_static["frame_times"] = torch.zeros_like(batch["frame_times"])
            out = self(batch_static)
            if "comp_rgb" in out:
                self.save_image_grid(
                    f"it{self.true_global_step}-test_static/{batch_static['index'][0]}.png",
                    (
                        [
                            {
                                "type": "rgb",
                                "img": out["comp_rgb"][0],
                                "kwargs": {"data_format": "HWC"},
                            },
                        ]
                    ),
                    name="test_step",
                    step=self.true_global_step
                )
            if "comp_normal" in out:
                self.save_image_grid(
                    f"it{self.true_global_step}-test-normal_static/{batch_static['index'][0]}.png",
                    (
                        [
                            {
                                "type": "rgb",
                                "img": out["comp_normal"][0],
                                "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                            }
                        ]
                    ),
                    name="test_step",
                    step=self.true_global_step
                )

    def on_test_epoch_end(self):
        fps = 15
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=fps,
            name=f"test",
            step=self.true_global_step,
        )
        if not self.static:
            self.save_img_sequence(
                    f"it{self.true_global_step}-test_static",
                    f"it{self.true_global_step}-test_static",
                    "(\d+)\.png",
                    save_format="mp4",
                    fps=fps,
                    name=f"test_static",
                    step=self.true_global_step,
                )
        out_depths = np.stack(self.out_depths)
        non_zeros_depth = out_depths[out_depths != 0]
        self.visu_perc_min_depth = np.percentile(non_zeros_depth, self.cfg.eval_depth_range_perc[0])
        self.visu_perc_max_depth = np.percentile(non_zeros_depth, self.cfg.eval_depth_range_perc[1])
        depth_color_maps = ['jet']
        for depth_color_map in depth_color_maps:
            for i, depth in enumerate(out_depths):
                self.save_image_grid(
                    f"it{self.true_global_step}-test-depth-{depth_color_map}/{i}.png",
                    [
                        {
                            "type": "grayscale",
                            "img": depth,
                            "kwargs": {"cmap": depth_color_map, "data_range": 'nonzero'},
                        },
                    ],
                    name="depth_test_step",
                    step=self.true_global_step,
                )
        extra_renderings = [f'depth-{depth_color_map}' for depth_color_map in depth_color_maps]
        for extra_rendering in extra_renderings:
            self.save_img_sequence(
                f"it{self.true_global_step}-test-{extra_rendering}",
                f"it{self.true_global_step}-test-{extra_rendering}",
                "(\d+)\.png",
                save_format="mp4",
                fps=fps,
                name=f"test",
                step=self.true_global_step,
            )
