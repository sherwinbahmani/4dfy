from dataclasses import dataclass

import threestudio
import torch
import torch.nn.functional as F
from threestudio.models.background.base import BaseBackground
from threestudio.models.geometry.base import BaseImplicitGeometry
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.renderers.base import VolumeRenderer
from threestudio.utils.typing import *
from copy import copy

@threestudio.register("stable-nerf-volume-renderer")
class PatchRenderer(VolumeRenderer):
    @dataclass
    class Config(VolumeRenderer.Config):
        base_renderer_type: str = ""
        base_renderer: Optional[VolumeRenderer.Config] = None
        mode: str = "interval"  # interval
        block_nums: Tuple = (3, 3)

    cfg: Config

    def configure(
        self,
        geometry: BaseImplicitGeometry,
        material: BaseMaterial,
        background: BaseBackground,
    ) -> None:
        if geometry.cfg.normal_type == "analytic":
            raise NotImplementedError(
                "Stable nerf renderer can not use analytic normal type."
            )
        if hasattr(background.cfg, "random_aug") and background.cfg.random_aug:
            raise NotImplementedError(
                "Stable nerf renderer does not support background augmentation."
            )
        self.base_renderer = threestudio.find(self.cfg.base_renderer_type)(
            self.cfg.base_renderer,
            geometry=geometry,
            material=material,
            background=background,
        )

    def forward(
        self,
        rays_o: Float[Tensor, "B H W 3"],
        rays_d: Float[Tensor, "B H W 3"],
        light_positions: Float[Tensor, "B 3"],
        bg_color: Optional[Tensor] = None,
        **kwargs,
    ) -> Dict[str, Float[Tensor, "..."]]:
        B, H, W, _ = rays_o.shape

        valid_patch_key = []
        out_global = {}
        if self.base_renderer.training:
            if self.cfg.mode == "interval":
                YI = self.cfg.block_nums[0]
                XI = self.cfg.block_nums[1]

                MAX_N_STATIC = copy(self.base_renderer.cfg.train_max_nums_static)
                MAX_N_DYNAMIC = copy(self.base_renderer.cfg.train_max_nums)
                if kwargs['is_video']:
                    MAX_N = copy(MAX_N_DYNAMIC)
                else:
                    MAX_N = copy(MAX_N_STATIC)
                for i in range(YI):
                    for j in range(XI):
                        int_rays_o = rays_o[:, i::YI, j::XI]
                        int_rays_d = rays_d[:, i::YI, j::XI]
                        if MAX_N > 0:
                            self.base_renderer.cfg.train_max_nums = MAX_N // (
                                YI * XI * B
                            )
                        out = self.base_renderer(
                            int_rays_o, int_rays_d, light_positions, bg_color, **kwargs
                        )
                        if len(valid_patch_key) == 0:
                            for key in out:
                                if torch.is_tensor(out[key]):
                                    if len(out[key].shape) == len(
                                        out["comp_rgb"].shape
                                    ):
                                        if (
                                            out[key][..., 0].shape
                                            == out["comp_rgb"][..., 0].shape
                                        ):
                                            valid_patch_key.append(key)
                                            out_global[key] = torch.zeros(
                                                B,
                                                H,
                                                W,
                                                out[key].shape[-1],
                                                dtype=out[key].dtype,
                                                device=out[key].device,
                                            )
                                    if len(out[key].shape) == len(out["weights"].shape):
                                        if (
                                            out[key][..., 0].shape
                                            == out["weights"][..., 0].shape
                                        ):
                                            valid_patch_key.append(key)
                                            out_global[key] = []
                        for key in valid_patch_key:
                            if torch.is_tensor(out_global[key]):
                                out_global[key][:, i::YI, j::XI] = out[key]
                            else:
                                out_global[key].append(out[key])
                for key in valid_patch_key:
                    if not torch.is_tensor(out_global[key]):
                        out_global[key] = torch.cat(out_global[key], dim=0)
                self.base_renderer.cfg.train_max_nums_static = MAX_N_STATIC
                self.base_renderer.cfg.train_max_nums = MAX_N_DYNAMIC
                out = out_global
        else:
            out = self.base_renderer(
                rays_o, rays_d, light_positions, bg_color, **kwargs
            )

        return out

    def update_step(
        self, epoch: int, global_step: int, on_load_weights: bool = False
    ) -> None:
        self.base_renderer.update_step(epoch, global_step, on_load_weights)

    def train(self, mode=True):
        return self.base_renderer.train(mode)

    def eval(self):
        return self.base_renderer.eval()
