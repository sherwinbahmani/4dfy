import bisect
import math
import random
from dataclasses import dataclass, field

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset

import threestudio
from threestudio import register
from threestudio.utils.base import Updateable
from threestudio.utils.config import parse_structured
from threestudio.utils.misc import get_device
from threestudio.utils.ops import (
    get_mvp_matrix,
    get_projection_matrix,
    get_ray_directions,
    get_rays,
)
from threestudio.utils.typing import *


@dataclass
class RandomCameraDataModuleConfig:
    # height and width should be Union[int, List[int]]
    # but OmegaConf does not support Union of containers
    height: Any = 64
    width: Any = 64
    resolution_milestones: List[int] = field(default_factory=lambda: [])
    eval_height: int = 512
    eval_width: int = 512
    batch_size: int = 1
    eval_batch_size: int = 1
    n_val_views: int = 1
    n_test_views: int = 120
    elevation_range: Tuple[float, float] = (-10, 90)
    azimuth_range: Tuple[float, float] = (-180, 180)
    camera_distance_range: Tuple[float, float] = (1, 1.5)
    fovy_range: Tuple[float, float] = (
        40,
        70,
    )  # in degrees, in vertical direction (along height)
    camera_perturb: float = 0.1
    center_perturb: float = 0.2
    up_perturb: float = 0.02
    light_position_perturb: float = 1.0
    light_distance_range: Tuple[float, float] = (0.8, 1.5)
    eval_elevation_deg: float = 15.0
    eval_camera_distance: float = 1.5
    eval_fovy_deg: float = 70.0
    light_sample_strategy: str = "dreamfusion"
    batch_uniform_azimuth: bool = True
    # Dynamic
    static: bool = True
    num_frames: int = 1
    sample_rand_frames: Optional[str] = None
    # Simultaneous training
    simultan: bool = False
    prob_single_view_video: Optional[float] = None
    width_vid: int = 64
    height_vid: int = 64
    num_frames_factor: int = 1
    train_dynamic_camera: Optional[bool] = False
    num_test_loop_factor: int = 1
    num_test_loop_static: int = 4
    test_traj: Optional[str] = None

class RandomCameraIterableDataset(IterableDataset, Updateable):
    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.cfg: RandomCameraDataModuleConfig = cfg
        self.heights: List[int] = (
            [self.cfg.height] if isinstance(self.cfg.height, int) else self.cfg.height
        )
        self.widths: List[int] = (
            [self.cfg.width] if isinstance(self.cfg.width, int) else self.cfg.width
        )
        self.batch_sizes: List[int] = (
            [self.cfg.batch_size]
            if isinstance(self.cfg.batch_size, int)
            else self.cfg.batch_size
        )
        if (
            len(self.heights) != 1
            and len(self.widths) != 1
            and len(self.batch_sizes) == 1
        ):
            self.batch_sizes = self.batch_sizes*len(self.heights)
        assert len(self.heights) == len(self.widths) == len(self.batch_sizes)
        self.resolution_milestones: List[int]
        if (
            len(self.heights) == 1
            and len(self.widths) == 1
            and len(self.batch_sizes) == 1
        ):
            if len(self.cfg.resolution_milestones) > 0:
                threestudio.warn(
                    "Ignoring resolution_milestones since height and width are not changing"
                )
            self.resolution_milestones = [-1]
        else:
            assert len(self.heights) == len(self.cfg.resolution_milestones) + 1
            self.resolution_milestones = [-1] + self.cfg.resolution_milestones

        self.directions_unit_focals = [
            get_ray_directions(H=height, W=width, focal=1.0)
            for (height, width) in zip(self.heights, self.widths)
        ]
        self.height: int = self.heights[0]
        self.width: int = self.widths[0]
        self.batch_size: int = self.batch_sizes[0]
        self.directions_unit_focal = self.directions_unit_focals[0]
        self.elevation_range = self.cfg.elevation_range
        self.azimuth_range = self.cfg.azimuth_range
        self.camera_distance_range = self.cfg.camera_distance_range
        self.fovy_range = self.cfg.fovy_range
        self.simultan_idx = 0
        if self.cfg.simultan:
            self.directions_unit_focals_vid = get_ray_directions(H=self.cfg.height_vid, W=self.cfg.width_vid, focal=1.0)
            self.height_vid = self.cfg.height_vid
            self.width_vid = self.cfg.width_vid
        else:
            self.directions_unit_focals_vid = None
            self.height_vid = None
            self.width_vid = None
        if self.cfg.train_dynamic_camera:
            self.elevation_range_delta = (-180./(16*self.cfg.num_frames), 180./(16*self.cfg.num_frames))
            self.azimuth_range_delta = (-180./(2*self.cfg.num_frames), 180./(2*self.cfg.num_frames))
            self.camera_distance_range_delta = lambda x: (
                (self.camera_distance_range[0]-x)/self.cfg.num_frames, (self.camera_distance_range[1]-x)/self.cfg.num_frames
                )

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        size_ind = bisect.bisect_right(self.resolution_milestones, global_step) - 1
        self.height = self.heights[size_ind]
        self.width = self.widths[size_ind]
        self.batch_size = self.batch_sizes[size_ind]
        self.directions_unit_focal = self.directions_unit_focals[size_ind]
        threestudio.debug(f"Training height: {self.height}, width: {self.width}")

    def __iter__(self):
        while True:
            yield {}

    def collate(self, batch) -> Dict[str, Any]:
        # Simultaneous training
        if self.cfg.simultan:
            if self.cfg.prob_single_view_video is not None:
                is_video = random.random() < self.cfg.prob_single_view_video
            else:
                is_video = False
        else:
            is_video = False
        if is_video:
            height = self.height_vid
            width = self.width_vid
            is_video = True
            directions_unit_focal = self.directions_unit_focals_vid
        else:
            height = self.height
            width = self.width
            is_video = False
            directions_unit_focal = self.directions_unit_focal
        if self.cfg.simultan:
            self.simultan_idx += 1
        train_dynamic_camera = self.cfg.train_dynamic_camera and is_video
        batch_size = self.batch_size
        num_frames = self.cfg.num_frames
        if train_dynamic_camera:
            batch_factor = num_frames
        else:
            batch_factor = 1
        batch_size = batch_size * batch_factor 
        # sample elevation angles
        elevation_deg: Float[Tensor, "B"]
        elevation: Float[Tensor, "B"]
        if random.random() < 0.5:
            # sample elevation angles uniformly with a probability 0.5 (biased towards poles)
            elevation_deg = (
                torch.rand(self.batch_size).repeat(batch_factor)
                * (self.cfg.elevation_range[1] - self.cfg.elevation_range[0])
                + self.cfg.elevation_range[0]
            )
            elevation = elevation_deg * math.pi / 180
        else:
            # otherwise sample uniformly on sphere
            elevation_range_percent = [
                (self.cfg.elevation_range[0] + 90.0) / 180.0,
                (self.cfg.elevation_range[1] + 90.0) / 180.0,
            ]
            # inverse transform sampling
            elevation = torch.asin(
                2
                * (
                    torch.rand(self.batch_size).repeat(batch_factor)
                    * (elevation_range_percent[1] - elevation_range_percent[0])
                    + elevation_range_percent[0]
                )
                - 1.0
            )
            elevation_deg = elevation / math.pi * 180.0
        if train_dynamic_camera:
            elevation_delta_deg = (
                torch.rand(self.batch_size)
                * (self.elevation_range_delta[1] - self.elevation_range_delta[0])
                + self.elevation_range_delta[0]
            ) * torch.arange(num_frames)
            elevation_deg = elevation_deg + elevation_delta_deg
            elevation = elevation + elevation_delta_deg * math.pi / 180
        # sample azimuth angles from a uniform distribution bounded by azimuth_range
        azimuth_deg: Float[Tensor, "B"]
        if self.cfg.batch_uniform_azimuth:
            assert batch_factor == 1
            # ensures sampled azimuth angles in a batch cover the whole range
            azimuth_deg = (
                torch.rand(self.batch_size) + torch.arange(self.batch_size)
            ) / self.batch_size * (
                self.cfg.azimuth_range[1] - self.cfg.azimuth_range[0]
            ) + self.cfg.azimuth_range[
                0
            ]
        else:
            # simple random sampling
            azimuth_deg = (
                torch.rand(self.batch_size).repeat(batch_factor)
                * (self.cfg.azimuth_range[1] - self.cfg.azimuth_range[0])
                + self.cfg.azimuth_range[0]
            )
        azimuth = azimuth_deg * math.pi / 180
        if train_dynamic_camera:
            azimuth_delta_deg = (
                torch.rand(self.batch_size)
                * (self.azimuth_range_delta[1] - self.azimuth_range_delta[0])
                + self.azimuth_range_delta[0]
            ) * torch.arange(num_frames)
            azimuth_deg = azimuth_deg + azimuth_delta_deg
            azimuth = azimuth + elevation_delta_deg * math.pi / 180

        # sample distances from a uniform distribution bounded by distance_range
        camera_distances: Float[Tensor, "B"] = (
            torch.rand(self.batch_size).repeat(batch_factor)
            * (self.cfg.camera_distance_range[1] - self.cfg.camera_distance_range[0])
            + self.cfg.camera_distance_range[0]
        )
        if train_dynamic_camera:
            camera_distance_range_delta = self.camera_distance_range_delta(camera_distances[0])
            camera_distance_delta = (
                torch.rand(self.batch_size)
                * (camera_distance_range_delta[1] - camera_distance_range_delta[0])
                + camera_distance_range_delta[0]
            ) * torch.arange(num_frames)
            camera_distances = camera_distances + camera_distance_delta

        # convert spherical coordinates to cartesian coordinates
        # right hand coordinate system, x back, y right, z up
        # elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
        camera_positions: Float[Tensor, "B 3"] = torch.stack(
            [
                camera_distances * torch.cos(elevation) * torch.cos(azimuth),
                camera_distances * torch.cos(elevation) * torch.sin(azimuth),
                camera_distances * torch.sin(elevation),
            ],
            dim=-1,
        )

        # default scene center at origin
        center: Float[Tensor, "B 3"] = torch.zeros_like(camera_positions)
        # default camera up direction as +z
        up: Float[Tensor, "B 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
            None, :
        ].repeat(batch_size, 1)

        # sample camera perturbations from a uniform distribution [-camera_perturb, camera_perturb]
        camera_perturb: Float[Tensor, "B 3"] = (
            torch.rand(self.batch_size, 3) * 2 * self.cfg.camera_perturb
            - self.cfg.camera_perturb
        ).repeat(batch_factor, 1)
        camera_positions = camera_positions + camera_perturb
        # sample center perturbations from a normal distribution with mean 0 and std center_perturb
        center_perturb: Float[Tensor, "B 3"] = (
            torch.randn(self.batch_size, 3) * self.cfg.center_perturb
        ).repeat(batch_factor, 1)
        center = center + center_perturb
        # sample up perturbations from a normal distribution with mean 0 and std up_perturb
        up_perturb: Float[Tensor, "B 3"] = (
            torch.randn(self.batch_size, 3) * self.cfg.up_perturb
        ).repeat(batch_factor, 1)
        up = up + up_perturb

        # sample fovs from a uniform distribution bounded by fov_range
        fovy_deg: Float[Tensor, "B"] = (
            torch.rand(self.batch_size)
            * (self.cfg.fovy_range[1] - self.cfg.fovy_range[0])
            + self.cfg.fovy_range[0]
        ).repeat(batch_factor)
        fovy = fovy_deg * math.pi / 180

        # sample light distance from a uniform distribution bounded by light_distance_range
        light_distances: Float[Tensor, "B"] = (
            torch.rand(self.batch_size)
            * (self.cfg.light_distance_range[1] - self.cfg.light_distance_range[0])
            + self.cfg.light_distance_range[0]
        ).repeat(batch_factor)

        if self.cfg.light_sample_strategy == "dreamfusion":
            # sample light direction from a normal distribution with mean camera_position and std light_position_perturb
            light_direction: Float[Tensor, "B 3"] = F.normalize(
                camera_positions
                + torch.randn(self.batch_size, 3).repeat(batch_factor, 1) * self.cfg.light_position_perturb,
                dim=-1,
            )
            # get light position by scaling light direction by light distance
            light_positions: Float[Tensor, "B 3"] = (
                light_direction * light_distances[:, None]
            )
        elif self.cfg.light_sample_strategy == "magic3d":
            # sample light direction within restricted angle range (pi/3)
            local_z = F.normalize(camera_positions, dim=-1)
            local_x = F.normalize(
                torch.stack(
                    [local_z[:, 1], -local_z[:, 0], torch.zeros_like(local_z[:, 0])],
                    dim=-1,
                ),
                dim=-1,
            )
            local_y = F.normalize(torch.cross(local_z, local_x, dim=-1), dim=-1)
            rot = torch.stack([local_x, local_y, local_z], dim=-1)
            light_azimuth = (
                torch.rand(self.batch_size) * math.pi - 2 * math.pi
            ).repeat(batch_factor)  # [-pi, pi]
            light_elevation = (
                torch.rand(self.batch_size) * math.pi / 3 + math.pi / 6
            ).repeat(batch_factor)  # [pi/6, pi/2]
            light_positions_local = torch.stack(
                [
                    light_distances
                    * torch.cos(light_elevation)
                    * torch.cos(light_azimuth),
                    light_distances
                    * torch.cos(light_elevation)
                    * torch.sin(light_azimuth),
                    light_distances * torch.sin(light_elevation),
                ],
                dim=-1,
            )
            light_positions = (rot @ light_positions_local[:, :, None])[:, :, 0]
        else:
            raise ValueError(
                f"Unknown light sample strategy: {self.cfg.light_sample_strategy}"
            )

        lookat: Float[Tensor, "B 3"] = F.normalize(center - camera_positions, dim=-1)
        right: Float[Tensor, "B 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        c2w3x4: Float[Tensor, "B 3 4"] = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
            dim=-1,
        )
        c2w: Float[Tensor, "B 4 4"] = torch.cat(
            [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
        )
        c2w[:, 3, 3] = 1.0

        # get directions by dividing directions_unit_focal by focal length
        focal_length: Float[Tensor, "B"] = 0.5 * height / torch.tan(0.5 * fovy)
        directions: Float[Tensor, "B H W 3"] = directions_unit_focal[
            None, :, :, :
        ].repeat(batch_size, 1, 1, 1)
        directions[:, :, :, :2] = (
            directions[:, :, :, :2] / focal_length[:, None, None, None]
        )

        # Importance note: the returned rays_d MUST be normalized!
        rays_o, rays_d = get_rays(directions, c2w, keepdim=True)

        proj_mtx: Float[Tensor, "B 4 4"] = get_projection_matrix(
            fovy, width / height, 0.1, 1000.0
        )  # FIXME: hard-coded near and far
        mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(c2w, proj_mtx)

        # Dynamic
        if self.cfg.sample_rand_frames == "t0":
            t0 = torch.FloatTensor(1).uniform_(0, 1/num_frames).item()
            frame_times = torch.linspace(t0, t0+(num_frames-1)/num_frames, num_frames)
        else:
            frame_times = torch.linspace(0.0, 1.0, num_frames)
        return {
            "rays_o": rays_o,
            "rays_d": rays_d,
            "mvp_mtx": mvp_mtx,
            "camera_positions": camera_positions,
            "c2w": c2w,
            "light_positions": light_positions,
            "elevation": elevation_deg,
            "azimuth": azimuth_deg,
            "camera_distances": camera_distances,
            "height": height,
            "width": width,
            "frame_times": frame_times,
            "frame_times_video": frame_times,
            "is_video": is_video,
            "train_dynamic_camera": train_dynamic_camera,
        }


class RandomCameraDataset(Dataset):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.cfg: RandomCameraDataModuleConfig = cfg
        self.split = split

        if split == "val":
            self.n_views = self.cfg.n_val_views
        else:
            self.n_views = self.cfg.n_test_views
        
        num_frames = self.cfg.num_frames
        if self.cfg.static:
            n_views_azimuth = self.n_views
        else:
            if split in ["val", "test"] and not self.cfg.static:
                num_frames = num_frames*self.cfg.num_frames_factor
            if self.split == "test":
                if self.cfg.test_traj in ['motion_smooth', 'motion_smooth_full']:
                    self.n_views = num_frames*2
                    n_views_azimuth = self.n_views//8
                    if self.cfg.test_traj == 'motion_smooth':
                        self.n_views = self.n_views-4
                    num_frames_static = n_views_azimuth
                else:
                    n_views_azimuth = num_frames*self.cfg.num_test_loop_factor
                    self.n_views = 2*n_views_azimuth*self.cfg.num_test_loop_static
                    num_frames_static = num_frames
            elif self.split == "val":
                n_views_azimuth = self.n_views

        azimuth_deg: Float[Tensor, "B"]
        if self.split == "val":
            # make sure the first and last view are not the same
            azimuth_deg = torch.linspace(0, 360.0, n_views_azimuth + 1)[: n_views_azimuth]
        # else:
            # azimuth_deg = torch.linspace(0, 360.0, n_views_azimuth
            # )
        elif self.split == "test":
            if self.cfg.static:
                azimuth_deg = torch.linspace(0, 360.0, n_views_azimuth)
            else:
                assert n_views_azimuth % self.cfg.num_test_loop_static == 0
                azimuth_deg = []
                for i in range(self.cfg.num_test_loop_static):
                    if self.cfg.test_traj in ['motion_smooth', 'motion_smooth_full']:
                        azimuth_start = (self.cfg.num_test_loop_static-i)*90.
                        azimuth_end = (self.cfg.num_test_loop_static-i-1)*90.
                    else:
                        azimuth_start = i*90.
                        azimuth_end = (i+1)*90.
                    azimuth_static_deg_i = torch.full((num_frames_static,), azimuth_start)
                    azimuth_deg.append(azimuth_static_deg_i)
                    azimuth_dynamic_deg_i = torch.linspace(azimuth_start, azimuth_end, n_views_azimuth)
                    if self.cfg.test_traj == 'motion_smooth':
                        azimuth_dynamic_deg_i = azimuth_dynamic_deg_i[1:]
                    azimuth_deg.append(azimuth_dynamic_deg_i)
                azimuth_deg = torch.cat(azimuth_deg)
        elevation_deg: Float[Tensor, "B"] = torch.full_like(
            azimuth_deg, self.cfg.eval_elevation_deg
        )
        camera_distances: Float[Tensor, "B"] = torch.full_like(
            elevation_deg, self.cfg.eval_camera_distance
        )

        elevation = elevation_deg * math.pi / 180
        azimuth = azimuth_deg * math.pi / 180

        # convert spherical coordinates to cartesian coordinates
        # right hand coordinate system, x back, y right, z up
        # elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
        camera_positions: Float[Tensor, "B 3"] = torch.stack(
            [
                camera_distances * torch.cos(elevation) * torch.cos(azimuth),
                camera_distances * torch.cos(elevation) * torch.sin(azimuth),
                camera_distances * torch.sin(elevation),
            ],
            dim=-1,
        )

        # default scene center at origin
        center: Float[Tensor, "B 3"] = torch.zeros_like(camera_positions)
        # default camera up direction as +z
        up: Float[Tensor, "B 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
            None, :
        ].repeat(self.cfg.eval_batch_size, 1)

        fovy_deg: Float[Tensor, "B"] = torch.full_like(
            elevation_deg, self.cfg.eval_fovy_deg
        )
        fovy = fovy_deg * math.pi / 180
        light_positions: Float[Tensor, "B 3"] = camera_positions

        lookat: Float[Tensor, "B 3"] = F.normalize(center - camera_positions, dim=-1)
        right: Float[Tensor, "B 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        c2w3x4: Float[Tensor, "B 3 4"] = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
            dim=-1,
        )
        c2w: Float[Tensor, "B 4 4"] = torch.cat(
            [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
        )
        c2w[:, 3, 3] = 1.0

        # get directions by dividing directions_unit_focal by focal length
        focal_length: Float[Tensor, "B"] = (
            0.5 * self.cfg.eval_height / torch.tan(0.5 * fovy)
        )
        directions_unit_focal = get_ray_directions(
            H=self.cfg.eval_height, W=self.cfg.eval_width, focal=1.0
        )
        directions: Float[Tensor, "B H W 3"] = directions_unit_focal[
            None, :, :, :
        ].repeat(self.n_views, 1, 1, 1)
        directions[:, :, :, :2] = (
            directions[:, :, :, :2] / focal_length[:, None, None, None]
        )

        rays_o, rays_d = get_rays(directions, c2w, keepdim=True)
        proj_mtx: Float[Tensor, "B 4 4"] = get_projection_matrix(
            fovy, self.cfg.eval_width / self.cfg.eval_height, 0.1, 1000.0
        )  # FIXME: hard-coded near and far
        mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(c2w, proj_mtx)

        if self.cfg.test_traj in ['motion_smooth', 'motion_smooth_full']:
            frame_times = torch.linspace(0, 1.0, num_frames)
            if self.cfg.test_traj == 'motion_smooth':
                frame_times = frame_times[:num_frames-2]
            else:
                frame_times = frame_times[:num_frames]
            frame_times = frame_times.repeat(math.ceil(self.n_views/num_frames))
        else:
            frame_times = torch.linspace(
            0, 1.0, num_frames
            ).repeat(math.ceil(self.n_views/num_frames))
            frame_times = frame_times[:self.n_views]
        frame_times_video = torch.linspace(
            0, 1.0, num_frames
            )
        self.rays_o, self.rays_d = rays_o, rays_d
        self.mvp_mtx = mvp_mtx
        self.c2w = c2w
        self.camera_positions = camera_positions
        self.light_positions = light_positions
        self.elevation, self.azimuth = elevation, azimuth
        self.elevation_deg, self.azimuth_deg = elevation_deg, azimuth_deg
        self.camera_distances = camera_distances
        self.frame_times = frame_times
        self.frame_times_video = frame_times_video

    def __len__(self):
        return self.n_views

    def __getitem__(self, index):
        return {
            "index": index,
            "rays_o": self.rays_o[index],
            "rays_d": self.rays_d[index],
            "mvp_mtx": self.mvp_mtx[index],
            "c2w": self.c2w[index],
            "camera_positions": self.camera_positions[index],
            "light_positions": self.light_positions[index],
            "elevation": self.elevation_deg[index],
            "azimuth": self.azimuth_deg[index],
            "camera_distances": self.camera_distances[index],
            "frame_times": self.frame_times[[index]],
            "frame_times_video": self.frame_times_video,
            "train_dynamic_camera": False,
        }

    def collate(self, batch):
        batch = torch.utils.data.default_collate(batch)
        batch.update({"height": self.cfg.eval_height, "width": self.cfg.eval_width})
        return batch


@register("random-camera-datamodule")
class RandomCameraDataModule(pl.LightningDataModule):
    cfg: RandomCameraDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(RandomCameraDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = RandomCameraIterableDataset(self.cfg)
        if stage in [None, "fit", "validate"]:
            self.val_dataset = RandomCameraDataset(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = RandomCameraDataset(self.cfg, "test")

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size, collate_fn=None) -> DataLoader:
        return DataLoader(
            dataset,
            # very important to disable multi-processing if you want to change self attributes at runtime!
            # (for example setting self.width and self.height in update_step)
            num_workers=0,  # type: ignore
            batch_size=batch_size,
            collate_fn=collate_fn,
        )

    def train_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.train_dataset, batch_size=None, collate_fn=self.train_dataset.collate
        )

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.val_dataset, batch_size=1, collate_fn=self.val_dataset.collate
        )

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate
        )

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate
        )
