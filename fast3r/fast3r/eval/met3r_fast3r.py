import math
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Identity, Module
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    AlphaCompositor,
    PerspectiveCameras,
    PointsRasterizationSettings,
    PointsRasterizer,
)

from lpips import LPIPS

from fast3r.models.fast3r import Fast3R
from fast3r.models.multiview_dust3r_module import MultiViewDUSt3RLitModule

from met3r.met3r import mmd_rbf, DEFAULT_REFERENCE_SIGMA, REFERENCE_SIGMA_BY_K, get_reference_sigma, compute_reference_sigma, energy_distance, mmd_imq


def freeze_model(module: Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False
    module.eval()


def convert_to_buffer(module: Module, persistent: bool = True) -> None:
    for _, child in list(module.named_children()):
        convert_to_buffer(child, persistent)
    for name, parameter_or_buffer in (
        *module.named_parameters(recurse=False),
        *module.named_buffers(recurse=False),
    ):
        value = parameter_or_buffer.detach().clone()
        delattr(module, name)
        module.register_buffer(name, value, persistent=persistent)


class MEt3R_Fast3R(Module):
    """MEt3R-style self-consistency metric driven by Fast3R predictions."""

    def __init__(
        self,
        *,
        img_size: Optional[Tuple[int, int]] = None,
        distance: Literal["cosine", "mse", "rmse", "psnr", "lpips"] = "cosine",
        feature_backbone: Optional[Literal["dino16", "dinov2", "maskclip", "vit", "clip", "resnet50"]] = "dino16",
        feature_backbone_weights: Optional[str] = "mhamilton723/FeatUp",
        upsampler: Optional[Literal["featup", "nearest", "bilinear", "bicubic"]] = "featup",
        use_norm: bool = True,
        confidence_threshold: float = 0.0,
        freeze: bool = True,
        rasterizer_kwargs: Optional[Dict] = None,
        fast3r_weights: Optional[str] = "jedyang97/Fast3R_ViT_Large_512",
        fast3r_model: Optional[Fast3R] = None,
        focal_length_estimation_method: str = "first_view_from_global_head",
        pnp_iterations: int = 100,
        default_focal_px: float = 1.6,
        min_points_per_view: int = 50,
        depth_consistency_tol: float = 0.5,
        samples_per_pair: int = 2048,
        oversample_factor: int = 4,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        if dtype is None:
            dtype = (
                torch.bfloat16
                if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
                else torch.float16
            )
        self._dtype = dtype

        self.img_size = img_size
        self.distance = distance
        self._feature_backbone = feature_backbone
        self.upsampler = upsampler
        self.confidence_threshold = confidence_threshold
        self._rasterizer_kwargs = rasterizer_kwargs or {}
        self.focal_length_estimation_method = focal_length_estimation_method
        self.pnp_iterations = pnp_iterations
        self.default_focal_px = default_focal_px
        self.min_points_per_view = min_points_per_view
        self.depth_consistency_tol = depth_consistency_tol
        self.samples_per_pair = samples_per_pair
        self.oversample_factor = oversample_factor

        if self.upsampler == "featup" and self.distance != "cosine":
            raise ValueError("FeatUp upsampler is only supported with cosine distance.")

        # Feature backbone configuration (FeatUp or standard features)
        if distance == "cosine":
            if feature_backbone and feature_backbone_weights and "FeatUp" in feature_backbone_weights:
                from featup.util import norm as feat_norm

                self.norm = feat_norm
                featup = torch.hub.load(feature_backbone_weights, feature_backbone, use_norm=use_norm)
                self.feature_model = featup.model
                if upsampler == "featup":
                    self.upsampler_model = featup.upsampler
                    if freeze:
                        freeze_model(self.upsampler_model)
                        convert_to_buffer(self.upsampler_model, persistent=False)
            else:
                self.norm = Identity()
                self.feature_model = torch.hub.load(feature_backbone_weights, feature_backbone) if feature_backbone else Identity()
            if freeze and isinstance(self.feature_model, Module):
                freeze_model(self.feature_model)
                convert_to_buffer(self.feature_model, persistent=False)
        else:
            self.norm = Identity()
            self.feature_model = Identity()

        if distance == "lpips":
            self.lpips = LPIPS(spatial=True)

        # Fast3R backbone
        if fast3r_model is None:
            if fast3r_weights is None:
                raise ValueError("Either fast3r_model or fast3r_weights must be provided.")
            self.fast3r = Fast3R.from_pretrained(fast3r_weights)
        else:
            self.fast3r = fast3r_model
        self.fast3r = self.fast3r.to(self._device)
        if freeze:
            freeze_model(self.fast3r)

        # Rendering utilities
        self.compositor = AlphaCompositor()
        if self.img_size is not None:
            self.set_rasterizer(image_size=self.img_size, **self._rasterizer_kwargs)

    def set_rasterizer(
        self,
        *,
        image_size: Tuple[int, int],
        points_per_pixel: int = 10,
        bin_size: int = 0,
        radius: Optional[float] = None,
        **kwargs,
    ) -> None:
        if radius is None:
            radius = 0.01
        self.rasterizer = PointsRasterizer(
            cameras=None,
            raster_settings=PointsRasterizationSettings(
                image_size=image_size,
                points_per_pixel=points_per_pixel,
                bin_size=bin_size,
                radius=radius,
                **kwargs,
            ),
        )

    def render(
        self,
        point_clouds: Pointclouds,
        cameras: PerspectiveCameras,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        with torch.autocast("cuda", enabled=False):
            fragments = self.rasterizer(point_clouds, cameras=cameras)
        radius = self.rasterizer.raster_settings.radius
        dists = fragments.dists.permute(0, 3, 1, 2)
        weights = 1 - dists / (radius * radius)
        feats = point_clouds.features_packed().permute(1, 0)
        images = self.compositor(fragments.idx.long().permute(0, 3, 1, 2), weights, feats)
        images = images.permute(0, 2, 3, 1)
        valid = fragments.idx[..., 0] != -1
        images = images * valid.unsqueeze(-1).to(images.dtype)
        return images, valid, fragments.zbuf

    def _rasterize_ids(
        self,
        point_clouds: Pointclouds,
        cameras: PerspectiveCameras,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        with torch.autocast("cuda", enabled=False):
            fragments = self.rasterizer(point_clouds, cameras=cameras)
        idx = fragments.idx[0, ..., 0].long()
        depth = fragments.zbuf[0, ..., 0]
        valid = idx != -1
        return idx, depth, valid

    def _project_points(
        self,
        cameras: PerspectiveCameras,
        points_world: Tensor,
        eps: float = 1e-6,
    ) -> Tuple[Tensor, Tensor]:
        R = cameras.R[0]
        T = cameras.T[0]
        points_cam = points_world @ R.T + T
        z = points_cam[:, 2]
        xy = points_cam[:, :2] / (z[:, None] + eps)
        uv = xy * cameras.focal_length[0] + cameras.principal_point[0]
        return uv, z

    def _uv_to_grid(self, uv: Tensor, image_hw: Tuple[int, int]) -> Tensor:
        H, W = image_hw
        x = (2.0 * uv[:, 0] / max(W - 1, 1)) - 1.0
        y = (2.0 * uv[:, 1] / max(H - 1, 1)) - 1.0
        return torch.stack([x, y], dim=-1)

    def _sample_warp_errors(
        self,
        *,
        src_flat_feats: Tensor,
        src_idx_flat: Tensor,
        src_valid_lin: Tensor,
        tgt_feats: Tensor,
        tgt_depth: Tensor,
        tgt_valid: Tensor,
        points_world: Tensor,
        tgt_cam: PerspectiveCameras,
        image_hw: Tuple[int, int],
        max_samples: Optional[int],
        depth_tol: float,
        oversample_factor: int,
        eps: float = 1e-6,
    ) -> Tensor:
        if src_valid_lin.numel() == 0:
            return torch.empty((0,), device=src_flat_feats.device, dtype=torch.float32)

        if max_samples is None:
            pix_lin = src_valid_lin
        else:
            num_candidates = min(src_valid_lin.numel(), max_samples * oversample_factor)
            perm = torch.randperm(src_valid_lin.numel(), device=src_valid_lin.device)[:num_candidates]
            pix_lin = src_valid_lin[perm]

        pt_ids = src_idx_flat[pix_lin]
        pts = points_world[pt_ids]
        uv, z = self._project_points(tgt_cam, pts, eps=eps)

        H, W = image_hw
        in_bounds = (
            (uv[:, 0] >= 0.0)
            & (uv[:, 0] <= float(W - 1))
            & (uv[:, 1] >= 0.0)
            & (uv[:, 1] <= float(H - 1))
            & (z.abs() > eps)
        )
        if not in_bounds.any():
            return torch.empty((0,), device=src_flat_feats.device, dtype=torch.float32)

        pix_lin = pix_lin[in_bounds]
        uv = uv[in_bounds]
        z = z[in_bounds]

        grid = self._uv_to_grid(uv, image_hw).view(1, 1, -1, 2)

        depth_in = tgt_depth.view(1, 1, H, W)
        valid_in = tgt_valid.view(1, 1, H, W)
        depth_s = F.grid_sample(depth_in, grid, mode="nearest", padding_mode="zeros", align_corners=True)[0, 0, 0]
        valid_s = F.grid_sample(valid_in, grid, mode="nearest", padding_mode="zeros", align_corners=True)[0, 0, 0]
        depth_ok = (valid_s > 0.5) & ((z - depth_s).abs() <= depth_tol)
        if not depth_ok.any():
            return torch.empty((0,), device=src_flat_feats.device, dtype=torch.float32)

        pix_lin = pix_lin[depth_ok]
        grid = grid[:, :, depth_ok, :]

        tgt_s = F.grid_sample(
            tgt_feats.unsqueeze(0),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )[0, :, 0].T
        src_s = src_flat_feats[pix_lin]

        if self.distance == "cosine":
            num = (src_s * tgt_s).sum(dim=-1)
            den = torch.linalg.norm(src_s, dim=-1) * torch.linalg.norm(tgt_s, dim=-1) + eps
            dist = 1.0 - (num / den).clamp(min=-1.0, max=1.0)
        elif self.distance == "mse":
            dist = ((src_s - tgt_s) ** 2).mean(dim=-1)
        elif self.distance == "rmse":
            dist = ((src_s - tgt_s) ** 2).mean(dim=-1).sqrt()
        elif self.distance == "psnr":
            mse = ((src_s - tgt_s) ** 2).mean(dim=-1)
            dist = 20.0 * torch.log10(1.0 / (mse.sqrt() + eps))
        else:
            raise NotImplementedError(f"Distance '{self.distance}' is not supported for correspondence sampling.")

        if max_samples is not None and dist.numel() > max_samples:
            idx = torch.randperm(dist.numel(), device=dist.device)[:max_samples]
            dist = dist[idx]

        return dist.to(torch.float32)

    def _get_features(self, images_01: Tensor) -> Tensor:
        # Ensure spatial size is compatible with the patch size used by the
        # feature backbone (in particular DINOv2 with patch size 14).
        patch_multiple: Optional[int] = None
        if self._feature_backbone == "dinov2":
            patch_multiple = 14

        if patch_multiple is not None:
            _, _, h, w = images_01.shape
            pad_h = (patch_multiple - (h % patch_multiple)) % patch_multiple
            pad_w = (patch_multiple - (w % patch_multiple)) % patch_multiple
            if pad_h or pad_w:
                # Pad on the bottom and right; later interpolations bring features
                # back to the original resolution used by the metric.
                images_01 = F.pad(images_01, (0, pad_w, 0, pad_h))

        return self.feature_model(self.norm(images_01))

    def _distance(
        self,
        a: Tensor,
        b: Tensor,
        mask: Optional[Tensor] = None,
        eps: float = 1e-6,
    ) -> Tuple[Tensor, Tensor]:
        if self.distance == "cosine":
            num = (a * b).sum(1)
            den = torch.linalg.norm(a, dim=1) * torch.linalg.norm(b, dim=1) + eps
            score_map = 1 - (num / den).clamp(min=-1.0, max=1.0)
            score_map = score_map[:, None]
        elif self.distance == "mse":
            score_map = ((a - b) ** 2).mean(1, keepdim=True)
        elif self.distance == "rmse":
            score_map = ((a - b) ** 2).mean(1, keepdim=True).sqrt()
        elif self.distance == "psnr":
            mse = F.mse_loss(a, b, reduction="none").mean(1, keepdim=True)
            score_map = 20 * torch.log10(1.0 / (mse.sqrt() + eps))
        elif self.distance == "lpips":
            score_map = self.lpips(2 * a - 1, 2 * b - 1)[:, None]
        else:
            raise NotImplementedError(self.distance)

        if mask is None:
            weighted = score_map.mean(dim=(2, 3))
        else:
            weighted = (score_map * mask[:, None]).sum(dim=(2, 3)) / (mask.sum(dim=(1, 2))[:, None] + eps)
        return score_map[:, 0], weighted[:, 0]

    def _prepare_views(self, images_01: Tensor) -> List[Dict[str, Tensor]]:
        K, _, H, W = images_01.shape
        images_norm = (images_01 * 2.0) - 1.0
        views: List[Dict[str, Tensor]] = []
        for idx in range(K):
            view: Dict[str, Tensor] = {
                "img": images_norm[idx : idx + 1].to(self._device, dtype=torch.float32),
                "true_shape": torch.tensor([[H, W]], device=self._device, dtype=torch.long),
            }
            # populate optional metadata expected by helpers when debugging
            view["dataset"] = torch.tensor([0], device=self._device)
            view["label"] = torch.tensor([0], device=self._device)
            view["instance"] = torch.tensor([idx], device=self._device)
            views.append(view)
        return views

    def _run_fast3r(self, views: List[Dict[str, Tensor]]) -> List[Dict[str, Tensor]]:
        preds = self.fast3r(views)
        return preds

    def _resize_features(
        self,
        features: Tensor,
        target_hw: Tuple[int, int],
        ref: Optional[Tensor] = None,
    ) -> Tensor:
        if features.shape[-2:] == target_hw:
            return features
        if self.upsampler == "featup" and hasattr(self, "upsampler_model"):
            if ref is None:
                raise ValueError("Reference images required for FeatUp upsampling")
            return self.upsampler_model(features, ref)

        mode = self.upsampler if self.upsampler in {"nearest", "bilinear", "bicubic"} else "bilinear"
        if mode in {"bilinear", "bicubic"}:
            return F.interpolate(features, size=target_hw, mode=mode, align_corners=False)
        return F.interpolate(features, size=target_hw, mode=mode)

    def _build_pointcloud(
        self,
        preds: Sequence[Dict[str, Tensor]],
        per_view_feats: Tensor,
    ) -> Pointclouds:
        points_list: List[Tensor] = []
        feats_list: List[Tensor] = []
        for view_idx, pred in enumerate(preds):
            pts = pred["pts3d_in_other_view"][0]
            conf = pred.get("conf")
            conf_map = conf[0] if isinstance(conf, Tensor) else None

            pts_flat = pts.view(-1, 3)
            feats = per_view_feats[view_idx].permute(1, 2, 0).reshape(-1, per_view_feats.shape[1])

            if conf_map is not None:
                conf_flat = conf_map.reshape(-1)
                mask = conf_flat >= self.confidence_threshold
                if mask.sum() < self.min_points_per_view:
                    mask = conf_flat > 0
            else:
                mask = torch.ones(pts_flat.shape[0], dtype=torch.bool, device=pts_flat.device)

            if mask.sum() == 0:
                mask = torch.ones_like(mask)

            points_list.append(pts_flat[mask])
            feats_list.append(feats[mask])

        pts_cat = torch.cat(points_list, dim=0)
        feats_cat = torch.cat(feats_list, dim=0)
        return Pointclouds(points=[pts_cat], features=[feats_cat])

    def _poses_to_cameras(
        self,
        poses_c2w: Sequence[Tensor],
        focals: Sequence[Optional[float]],
        image_hw: Tuple[int, int],
    ) -> List[PerspectiveCameras]:
        H, W = image_hw
        cameras: List[PerspectiveCameras] = []
        for pose, focal in zip(poses_c2w, focals):
            pose_tensor = torch.as_tensor(pose, device=self._device, dtype=torch.float32)
            R_c2w = pose_tensor[:3, :3]
            t_c2w = pose_tensor[:3, 3]
            R = R_c2w.transpose(0, 1)
            T = -R @ t_c2w
            focal_px = float(focal) if focal and math.isfinite(focal) else self.default_focal_px * max(H, W)
            fx = fy = focal_px
            cx = (W - 1) / 2.0
            cy = (H - 1) / 2.0
            cameras.append(
                PerspectiveCameras(
                    device=self._device,
                    R=R.unsqueeze(0),
                    T=T.unsqueeze(0),
                    focal_length=torch.tensor([[fx, fy]], device=self._device, dtype=torch.float32),
                    principal_point=torch.tensor([[cx, cy]], device=self._device, dtype=torch.float32),
                    image_size=torch.tensor([[H, W]], device=self._device, dtype=torch.float32),
                    in_ndc=False,
                )
            )
        return cameras

    def _estimate_cameras(
        self,
        preds: Sequence[Dict[str, Tensor]],
    ) -> Tuple[List[Tensor], List[Optional[float]]]:
        with torch.no_grad():
            poses_c2w_all, focals_all = MultiViewDUSt3RLitModule.estimate_camera_poses(
                preds=preds,
                niter_PnP=self.pnp_iterations,
                focal_length_estimation_method=self.focal_length_estimation_method,
            )
        return poses_c2w_all[0], focals_all[0]

    def _prepare_warp_data(
        self,
        images: Tensor,
    ) -> Tuple[Tensor, List[PerspectiveCameras], Tensor, Tensor, Tensor, Tensor, List[Tensor]]:
        images = images.to(self._device, dtype=torch.float32)
        if images.ndim != 5:
            raise ValueError(f"Expected images with shape (B,K,C,H,W), got {tuple(images.shape)}")
        B, K, _, _, _ = images.shape
        if B != 1:
            raise ValueError("MEt3R_Fast3R currently supports batch size 1; process samples sequentially.")

        sample = images[0]
        views = self._prepare_views(sample)
        preds = self._run_fast3r(views)

        pts_hw = tuple(int(v) for v in preds[0]["pts3d_in_other_view"].shape[1:3])
        if self.img_size is None:
            self.set_rasterizer(image_size=pts_hw, **self._rasterizer_kwargs)

        if self.distance == "cosine":
            flat = sample.to(self._device)
            feats_lr = self._get_features(flat)
            feats_img_res = self._resize_features(
                feats_lr,
                flat.shape[-2:],
                ref=flat if self.upsampler == "featup" else None,
            )
        else:
            feats_img_res = sample.to(self._device)

        feats_hr = F.interpolate(
            feats_img_res,
            size=pts_hw,
            mode="bilinear",
            align_corners=False,
        ).to(torch.float32)
        if self.distance == "cosine":
            self._last_feature_norm_values = torch.linalg.norm(
                feats_hr.detach(),
                dim=1,
            ).reshape(-1).to("cpu")
        else:
            self._last_feature_norm_values = torch.empty(0, dtype=torch.float32)

        confidence_values: List[Tensor] = []
        for pred in preds:
            conf = pred.get("conf")
            if isinstance(conf, Tensor):
                confidence_values.append(conf.reshape(-1).detach().to("cpu", dtype=torch.float32))
        self._last_confidence_values = (
            torch.cat(confidence_values, dim=0)
            if confidence_values
            else torch.empty(0, dtype=torch.float32)
        )

        points_list: List[Tensor] = []
        view_ids_list: List[Tensor] = []
        for view_k, pred in enumerate(preds):
            pts = pred["pts3d_in_other_view"][0]
            conf = pred.get("conf")
            conf_map = conf[0] if isinstance(conf, Tensor) else None

            pts_flat = pts.view(-1, 3)
            if conf_map is not None:
                conf_flat = conf_map.reshape(-1)
                mask = conf_flat >= self.confidence_threshold
                if mask.sum() < self.min_points_per_view:
                    mask = conf_flat > 0
            else:
                mask = torch.ones(pts_flat.shape[0], dtype=torch.bool, device=pts_flat.device)

            if mask.sum() == 0:
                mask = torch.ones_like(mask)

            points_list.append(pts_flat[mask])
            view_ids_list.append(torch.full((int(mask.sum()),), view_k, dtype=torch.long, device=pts_flat.device))

        points_world = torch.cat(points_list, dim=0)
        self._last_point_view_ids = torch.cat(view_ids_list, dim=0)
        cloud = Pointclouds(points=[points_world])

        poses_c2w, focals = self._estimate_cameras(preds)
        cameras = self._poses_to_cameras(poses_c2w, focals, pts_hw)

        idx_maps: List[Tensor] = []
        depth_maps: List[Tensor] = []
        valid_maps: List[Tensor] = []
        valid_lin: List[Tensor] = []
        for cam in cameras:
            idx, depth, valid = self._rasterize_ids(cloud, cameras=cam)
            idx_maps.append(idx)
            depth_maps.append(depth)
            valid_maps.append(valid)
            valid_lin.append(torch.nonzero(idx.view(-1) != -1, as_tuple=False).squeeze(1))

        idx_tensor = torch.stack(idx_maps, dim=0)
        depth_tensor = torch.stack(depth_maps, dim=0)
        valid_tensor = torch.stack(valid_maps, dim=0).float()
        self._last_idx_maps = idx_tensor
        self._last_valid_maps = valid_tensor

        return points_world, cameras, feats_hr, idx_tensor, depth_tensor, valid_tensor, valid_lin

    @torch.no_grad()
    def forward(
        self,
        images: Tensor,
        *,
        return_overlap_mask: bool = False,
        return_score_map: bool = False,
        return_projections: bool = False,
    ) -> Tuple[Tensor, ...]:
        if images.ndim != 5:
            raise ValueError(f"Expected images with shape (B,K,C,H,W), got {tuple(images.shape)}")
        if images.shape[0] != 1:
            raise ValueError("MEt3R_Fast3R currently supports batch size 1; process samples sequentially.")
        if images.shape[1] < 2:
            return (torch.full((1,), float("nan"), device=self._device),)

        (
            points_world,
            cameras,
            feats,
            idx_maps,
            depth_maps,
            valid_maps,
            valid_lin,
        ) = self._prepare_warp_data(images)

        K, C, H, W = feats.shape
        flat_feats = feats.permute(0, 2, 3, 1).reshape(K, H * W, C)
        idx_flat = idx_maps.view(K, -1)
        per_dir = max(1, self.samples_per_pair // 2)

        scores: List[Tensor] = []
        pair_details: List[dict] = []
        n_pairs = 0
        n_zero_overlap = 0
        for i in range(K):
            for j in range(i + 1, K):
                n_pairs += 1
                dir_scores: List[Tensor] = []
                pair_residuals: List[Tensor] = []
                for src, tgt in ((i, j), (j, i)):
                    dist = self._sample_warp_errors(
                        src_flat_feats=flat_feats[src],
                        src_idx_flat=idx_flat[src],
                        src_valid_lin=valid_lin[src],
                        tgt_feats=feats[tgt],
                        tgt_depth=depth_maps[tgt],
                        tgt_valid=valid_maps[tgt],
                        points_world=points_world,
                        tgt_cam=cameras[tgt],
                        image_hw=(H, W),
                        max_samples=per_dir,
                        depth_tol=self.depth_consistency_tol,
                        oversample_factor=self.oversample_factor,
                    )
                    if dist.numel() > 0:
                        dir_scores.append(dist.mean())
                        pair_residuals.append(dist.detach())
                if dir_scores:
                    scores.append(torch.stack(dir_scores).mean())
                else:
                    n_zero_overlap += 1
                pair_details.append({
                    "i": i, "j": j,
                    "has_overlap": bool(dir_scores),
                    "n_correspondences": sum(r.numel() for r in pair_residuals),
                    "mean_residual": float(torch.cat(pair_residuals).mean()) if pair_residuals else float("nan"),
                    "residuals_cpu": torch.cat(pair_residuals).cpu() if pair_residuals else torch.empty(0),
                })

        self._last_pair_details = pair_details
        self._last_overlap_stats = {"n_pairs": n_pairs, "n_zero_overlap": n_zero_overlap}
        batch_score = torch.stack(scores).mean().view(1) if scores else torch.full((1,), float("nan"), device=self._device)
        outputs: List[Tensor | None] = [batch_score]
        if return_overlap_mask:
            outputs.append(None)
        if return_score_map:
            outputs.append(None)
        if return_projections:
            outputs.append(None)
        return tuple(outputs)


class MEt3R_Fast3R_MMD(MEt3R_Fast3R):
    """
    MMD-based aggregation of Fast3R-driven feature discrepancies, analogous to MEt3R_MMD.
    
    Args:
        reference_sigma: If provided, use this fixed kernel bandwidth instead of
            computing adaptively per sample. Recommended for calibration tasks where
            you want comparable scores across samples with different error magnitudes.
            Set to DEFAULT_REFERENCE_SIGMA (0.15) for cosine distance on MipNeRF360.
        use_reference_sigma: If True and reference_sigma is None, use DEFAULT_REFERENCE_SIGMA
            (or per-K sigma from REFERENCE_SIGMA_BY_K if use_per_k_sigma is also True).
        use_per_k_sigma: If True and use_reference_sigma is True, look up sigma from
            REFERENCE_SIGMA_BY_K based on the number of views K. Falls back to
            DEFAULT_REFERENCE_SIGMA if K not in the mapping.
    """

    def __init__(
        self,
        *args,
        per_pair_max_samples: Optional[int] = 128,
        max_total_samples: Optional[int] = 4096,
        sample_on_cpu: bool = True,
        reference_sigma: Optional[float] = None,
        use_reference_sigma: bool = False,
        use_per_k_sigma: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.per_pair_max_samples = per_pair_max_samples
        self.max_total_samples = max_total_samples
        self.sample_on_cpu = sample_on_cpu
        self.reference_sigma = reference_sigma
        self.use_reference_sigma = use_reference_sigma
        self.use_per_k_sigma = use_per_k_sigma

    @torch.no_grad()
    def forward(
        self,
        images: Tensor,
        *,
        gt_images: Optional[Tensor] = None,
        return_sigma: bool = False,
        **kwargs,
    ) -> Tuple[Tensor, ...]:
        """
        Compute an MMD² score over pooled per-pixel errors across all view pairs.

        If gt_images is provided, we additionally compute a two-sample MMD between
        the error distributions of `images` and `gt_images`, mirroring MEt3R_MMD.
        
        Args:
            reference_sigma: Override instance reference_sigma for this call.
            use_reference_sigma: Override instance use_reference_sigma for this call.
            use_per_k_sigma: Override instance use_per_k_sigma for this call.
        """
        device = images.device
        b, k, c, h, w = images.shape
        images = images.to(self._device, dtype=torch.float32)

        per_pair_max_samples = kwargs.pop("per_pair_max_samples", self.per_pair_max_samples)
        max_total_samples = kwargs.pop("max_total_samples", self.max_total_samples)
        sample_on_cpu = kwargs.pop("sample_on_cpu", self.sample_on_cpu)
        reference_sigma = kwargs.pop("reference_sigma", self.reference_sigma)
        use_reference_sigma = kwargs.pop("use_reference_sigma", self.use_reference_sigma)
        use_per_k_sigma = kwargs.pop("use_per_k_sigma", self.use_per_k_sigma)
        
        # Resolve sigma: explicit > per-K lookup > global default > adaptive
        if reference_sigma is None and use_reference_sigma:
            if use_per_k_sigma:
                reference_sigma = get_reference_sigma(k)
            else:
                reference_sigma = DEFAULT_REFERENCE_SIGMA

        def _collect_errors(batch: Tensor) -> tuple[List[Tensor], int, int]:
            if batch.ndim != 5:
                raise ValueError(f"Expected images with shape (B,K,C,H,W), got {tuple(batch.shape)}")
            if batch.shape[0] != 1:
                raise ValueError("MEt3R_Fast3R_MMD currently supports batch size 1; process samples sequentially.")

            (
                points_world,
                cameras,
                feats,
                idx_maps,
                depth_maps,
                valid_maps,
                valid_lin,
            ) = super(MEt3R_Fast3R_MMD, self)._prepare_warp_data(batch)

            K, C, H, W = feats.shape
            flat_feats = feats.permute(0, 2, 3, 1).reshape(K, H * W, C)
            idx_flat = idx_maps.view(K, -1)
            per_dir = None if per_pair_max_samples is None else max(1, per_pair_max_samples // 2)

            errors: List[Tensor] = []
            n_pairs = 0
            n_zero_overlap = 0
            for i in range(K):
                for j in range(i + 1, K):
                    n_pairs += 1
                    pair_has_overlap = False
                    for src, tgt in ((i, j), (j, i)):
                        dist = super(MEt3R_Fast3R_MMD, self)._sample_warp_errors(
                            src_flat_feats=flat_feats[src],
                            src_idx_flat=idx_flat[src],
                            src_valid_lin=valid_lin[src],
                            tgt_feats=feats[tgt],
                            tgt_depth=depth_maps[tgt],
                            tgt_valid=valid_maps[tgt],
                            points_world=points_world,
                            tgt_cam=cameras[tgt],
                            image_hw=(H, W),
                            max_samples=per_dir,
                            depth_tol=self.depth_consistency_tol,
                            oversample_factor=self.oversample_factor,
                        )
                        if dist.numel() == 0:
                            continue
                        pair_has_overlap = True
                        values = dist.detach()
                        if sample_on_cpu:
                            values = values.to("cpu")
                        errors.append(values)
                    if not pair_has_overlap:
                        n_zero_overlap += 1

            return errors, n_pairs, n_zero_overlap

        err_list_pred, n_pairs, n_zero = _collect_errors(images)
        self._last_overlap_stats = {"n_pairs": n_pairs, "n_zero_overlap": n_zero}
        err_list_gt: List[Tensor] = []

        if gt_images is not None:
            gt_batch = gt_images.to(self._device, dtype=torch.float32)
            err_list_gt, _, _ = _collect_errors(gt_batch)

        if not err_list_pred:
            nan = torch.tensor(float("nan"), device=device)
            if return_sigma:
                return (nan, nan)
            return (nan,)

        delta_pred = torch.cat(err_list_pred)
        if max_total_samples is not None and delta_pred.numel() > max_total_samples:
            idx = torch.randperm(delta_pred.numel(), device=delta_pred.device)[:max_total_samples]
            delta_pred = delta_pred[idx]

        # Use fixed sigma if provided, otherwise compute adaptively or use reference
        if return_sigma:
            mmd_pred, sigma_pred = mmd_rbf(
                delta_pred,
                sigma=reference_sigma,
                return_sigma=True,
                use_reference_sigma=use_reference_sigma,
            )
        else:
            mmd_pred = mmd_rbf(
                delta_pred,
                sigma=reference_sigma,
                use_reference_sigma=use_reference_sigma,
            )

        if not err_list_gt:
            if return_sigma:
                return (mmd_pred.to(device), sigma_pred.to(device))
            return (mmd_pred.to(device),)

        delta_gt = torch.cat(err_list_gt)
        if max_total_samples is not None and delta_gt.numel() > max_total_samples:
            idx = torch.randperm(delta_gt.numel(), device=delta_gt.device)[:max_total_samples]
            delta_gt = delta_gt[idx]

        # Use fixed sigma if provided, otherwise compute from pooled samples
        if reference_sigma is not None:
            sigma = reference_sigma
        elif use_reference_sigma:
            sigma = DEFAULT_REFERENCE_SIGMA
        else:
            with torch.no_grad():
                pooled = torch.cat([delta_pred, delta_gt])
                n_pool = min(512, pooled.numel())
                idx_pool = torch.randperm(pooled.numel(), device=pooled.device)[:n_pool]
                diff = (pooled[idx_pool][:, None] - pooled[idx_pool][None, :]).abs()
                sigma = diff[diff > 0].median().clamp(min=1e-6).item()

        n_p, n_g = delta_pred.numel(), delta_gt.numel()
        K_pp = torch.exp(-(delta_pred[:, None] - delta_pred[None, :]) ** 2 / (2 * sigma**2))
        K_gg = torch.exp(-(delta_gt[:, None] - delta_gt[None, :]) ** 2 / (2 * sigma**2))
        K_pg = torch.exp(-(delta_pred[:, None] - delta_gt[None, :]) ** 2 / (2 * sigma**2))

        mmd2 = (K_pp.sum() - n_p) / (n_p * (n_p - 1))
        mmd2 += (K_gg.sum() - n_g) / (n_g * (n_g - 1))
        mmd2 += -2 * K_pg.mean()

        if return_sigma:
            sigma_tensor = torch.tensor(sigma, device=delta_gt.device)
            return (mmd2.to(device), sigma_tensor.to(device))

        return (mmd2.to(device),)


class MEt3R_Fast3R_PointConsistency(MEt3R_Fast3R):
    def __init__(
        self,
        *args,
        max_points: int = 20000,
        points_chunk_size: int = 8192,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.max_points = max_points
        self.points_chunk_size = points_chunk_size

    @torch.no_grad()
    def _point_dispersions(self, images: Tensor) -> Tensor:
        if self.distance != "cosine":
            raise ValueError("Point-consistency metric currently supports cosine distance only.")
        if images.ndim != 5:
            raise ValueError(f"Expected images with shape (B,K,C,H,W), got {tuple(images.shape)}")
        if images.shape[0] != 1:
            raise ValueError("MEt3R_Fast3R_PointConsistency currently supports batch size 1; process samples sequentially.")

        (
            points_world,
            cameras,
            feats,
            _idx_maps,
            depth_maps,
            valid_maps,
            _valid_lin,
        ) = self._prepare_warp_data(images)

        K, C, H, W = feats.shape
        num_points_total = points_world.shape[0]
        num_points = min(int(self.max_points), int(num_points_total))
        if num_points < 1:
            return torch.empty((0,), device=points_world.device, dtype=torch.float32)

        perm = torch.randperm(num_points_total, device=points_world.device)[:num_points]
        pts = points_world[perm]

        sum_feats = torch.zeros((num_points, C), device=points_world.device, dtype=torch.float32)
        counts = torch.zeros((num_points,), device=points_world.device, dtype=torch.int32)

        eps = 1e-6
        for k in range(K):
            depth_k = depth_maps[k]
            valid_k = valid_maps[k]
            feats_k = feats[k]

            for start in range(0, num_points, self.points_chunk_size):
                end = min(start + self.points_chunk_size, num_points)
                pts_chunk = pts[start:end]

                uv, z = self._project_points(cameras[k], pts_chunk, eps=eps)
                in_bounds = (
                    (uv[:, 0] >= 0.0)
                    & (uv[:, 0] <= float(W - 1))
                    & (uv[:, 1] >= 0.0)
                    & (uv[:, 1] <= float(H - 1))
                    & (z.abs() > eps)
                )
                if not in_bounds.any():
                    continue

                idx_in = torch.nonzero(in_bounds, as_tuple=False).squeeze(1)
                uv_in = uv[idx_in]
                z_in = z[idx_in]

                grid_in = self._uv_to_grid(uv_in, (H, W)).view(1, 1, -1, 2)
                depth_in = depth_k.view(1, 1, H, W)
                valid_in = valid_k.view(1, 1, H, W)
                depth_s = F.grid_sample(depth_in, grid_in, mode="nearest", padding_mode="zeros", align_corners=True)[
                    0, 0, 0
                ]
                valid_s = F.grid_sample(valid_in, grid_in, mode="nearest", padding_mode="zeros", align_corners=True)[
                    0, 0, 0
                ]
                depth_ok = (valid_s > 0.5) & ((z_in - depth_s).abs() <= self.depth_consistency_tol)
                if not depth_ok.any():
                    continue

                idx_vis = idx_in[depth_ok]
                grid_vis = grid_in[:, :, depth_ok, :]
                feats_vis = F.grid_sample(
                    feats_k.unsqueeze(0),
                    grid_vis,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=True,
                )[0, :, 0].T
                feats_vis = feats_vis / (torch.linalg.norm(feats_vis, dim=-1, keepdim=True) + eps)
                feats_vis = feats_vis.to(torch.float32)

                global_idx = start + idx_vis
                sum_feats[global_idx] += feats_vis
                counts[global_idx] += 1

        keep = counts >= 2
        self._last_overlap_stats = {
            "n_pairs": K * (K - 1) // 2,
            "n_zero_overlap": -1,
            "n_points_sampled": num_points,
            "n_points_multi_view": int(keep.sum().item()),
        }
        if not keep.any():
            return torch.empty((0,), device=points_world.device, dtype=torch.float32)

        n = counts[keep].to(torch.float32)
        s = sum_feats[keep]
        sum_norm_sq = (s * s).sum(dim=-1)
        mean_pairwise_sim = (sum_norm_sq - n) / (n * (n - 1.0))
        disp = 1.0 - mean_pairwise_sim
        return disp.to(torch.float32)

    @torch.no_grad()
    def forward(self, images: Tensor, **_kwargs) -> Tuple[Tensor, ...]:
        disp = self._point_dispersions(images)
        if disp.numel() == 0:
            return (torch.tensor(float("nan"), device=images.device),)
        return (disp.mean().view(1).to(images.device),)


class MEt3R_Fast3R_PointConsistency_MMD(MEt3R_Fast3R_PointConsistency):
    """
    MMD-based aggregation of point-consistency dispersions.
    
    Args:
        reference_sigma: If provided, use this fixed kernel bandwidth instead of
            computing adaptively per sample. Recommended for calibration tasks.
        use_reference_sigma: If True and reference_sigma is None, use DEFAULT_REFERENCE_SIGMA
            (or per-K sigma from REFERENCE_SIGMA_BY_K if use_per_k_sigma is also True).
        use_per_k_sigma: If True and use_reference_sigma is True, look up sigma from
            REFERENCE_SIGMA_BY_K based on the number of views K.
    """
    def __init__(
        self,
        *args,
        max_total_samples: Optional[int] = 4096,
        sample_on_cpu: bool = True,
        reference_sigma: Optional[float] = None,
        use_reference_sigma: bool = False,
        use_per_k_sigma: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.max_total_samples = max_total_samples
        self.sample_on_cpu = sample_on_cpu
        self.reference_sigma = reference_sigma
        self.use_reference_sigma = use_reference_sigma
        self.use_per_k_sigma = use_per_k_sigma

    @torch.no_grad()
    def forward(
        self,
        images: Tensor,
        *,
        gt_images: Optional[Tensor] = None,
        return_sigma: bool = False,
        reference_sigma: Optional[float] = None,
        use_reference_sigma: Optional[bool] = None,
        use_per_k_sigma: Optional[bool] = None,
        **_kwargs,
    ) -> Tuple[Tensor, ...]:
        device = images.device
        k = images.shape[1] if images.ndim == 5 else None
        
        # Use instance defaults if not overridden
        if reference_sigma is None:
            reference_sigma = self.reference_sigma
        if use_reference_sigma is None:
            use_reference_sigma = self.use_reference_sigma
        if use_per_k_sigma is None:
            use_per_k_sigma = self.use_per_k_sigma
        
        # Resolve sigma: explicit > per-K lookup > global default > adaptive
        if reference_sigma is None and use_reference_sigma:
            if use_per_k_sigma and k is not None:
                reference_sigma = get_reference_sigma(k)
            else:
                reference_sigma = DEFAULT_REFERENCE_SIGMA

        delta_pred = self._point_dispersions(images)
        if delta_pred.numel() == 0:
            nan = torch.tensor(float("nan"), device=device)
            if return_sigma:
                return (nan, nan)
            return (nan,)

        if self.sample_on_cpu:
            delta_pred = delta_pred.to("cpu")
        if self.max_total_samples is not None and delta_pred.numel() > self.max_total_samples:
            idx = torch.randperm(delta_pred.numel(), device=delta_pred.device)[: self.max_total_samples]
            delta_pred = delta_pred[idx]

        # Use fixed sigma if provided, otherwise compute adaptively or use reference
        if return_sigma:
            mmd_pred, sigma_pred = mmd_rbf(
                delta_pred,
                sigma=reference_sigma,
                return_sigma=True,
                use_reference_sigma=use_reference_sigma,
            )
        else:
            mmd_pred = mmd_rbf(
                delta_pred,
                sigma=reference_sigma,
                use_reference_sigma=use_reference_sigma,
            )

        if gt_images is None:
            if return_sigma:
                return (mmd_pred.to(device), sigma_pred.to(device))
            return (mmd_pred.to(device),)

        delta_gt = self._point_dispersions(gt_images)
        if delta_gt.numel() == 0:
            if return_sigma:
                return (mmd_pred.to(device), sigma_pred.to(device))
            return (mmd_pred.to(device),)

        if self.sample_on_cpu:
            delta_gt = delta_gt.to("cpu")
        if self.max_total_samples is not None and delta_gt.numel() > self.max_total_samples:
            idx = torch.randperm(delta_gt.numel(), device=delta_gt.device)[: self.max_total_samples]
            delta_gt = delta_gt[idx]

        # Use fixed sigma if provided, otherwise compute from pooled samples
        if reference_sigma is not None:
            sigma = reference_sigma
        elif use_reference_sigma:
            sigma = DEFAULT_REFERENCE_SIGMA
        else:
            with torch.no_grad():
                pooled = torch.cat([delta_pred, delta_gt])
                n_pool = min(512, pooled.numel())
                idx_pool = torch.randperm(pooled.numel(), device=pooled.device)[:n_pool]
                diff = (pooled[idx_pool][:, None] - pooled[idx_pool][None, :]).abs()
                sigma = diff[diff > 0].median().clamp(min=1e-6).item()

        n_p, n_g = delta_pred.numel(), delta_gt.numel()
        K_pp = torch.exp(-(delta_pred[:, None] - delta_pred[None, :]) ** 2 / (2 * sigma**2))
        K_gg = torch.exp(-(delta_gt[:, None] - delta_gt[None, :]) ** 2 / (2 * sigma**2))
        K_pg = torch.exp(-(delta_pred[:, None] - delta_gt[None, :]) ** 2 / (2 * sigma**2))

        mmd2 = (K_pp.sum() - n_p) / (n_p * (n_p - 1))
        mmd2 += (K_gg.sum() - n_g) / (n_g * (n_g - 1))
        mmd2 += -2 * K_pg.mean()

        if return_sigma:
            sigma_tensor = torch.tensor(sigma, device=delta_gt.device)
            return (mmd2.to(device), sigma_tensor.to(device))
        return (mmd2.to(device),)


class MEt3R_Fast3R_Energy(MEt3R_Fast3R):
    """
    Energy distance variant of Fast3R-driven feature consistency metric.
    
    Energy distance is a kernel-free metric that doesn't require bandwidth tuning,
    making it more robust for comparing samples with different error magnitudes.
    """

    def __init__(
        self,
        *args,
        per_pair_max_samples: Optional[int] = 128,
        max_total_samples: Optional[int] = 4096,
        sample_on_cpu: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.per_pair_max_samples = per_pair_max_samples
        self.max_total_samples = max_total_samples
        self.sample_on_cpu = sample_on_cpu

    @torch.no_grad()
    def forward(self, images: Tensor, **kwargs) -> Tuple[Tensor, ...]:
        device = images.device
        images = images.to(self._device, dtype=torch.float32)

        per_pair_max_samples = kwargs.pop("per_pair_max_samples", self.per_pair_max_samples)
        max_total_samples = kwargs.pop("max_total_samples", self.max_total_samples)
        sample_on_cpu = kwargs.pop("sample_on_cpu", self.sample_on_cpu)

        if images.ndim != 5 or images.shape[0] != 1:
            raise ValueError("Expected images with shape (1,K,C,H,W)")

        (points_world, cameras, feats, idx_maps, depth_maps, valid_maps, valid_lin) = super()._prepare_warp_data(images)
        K, C, H, W = feats.shape
        flat_feats = feats.permute(0, 2, 3, 1).reshape(K, H * W, C)
        idx_flat = idx_maps.view(K, -1)
        per_dir = None if per_pair_max_samples is None else max(1, per_pair_max_samples // 2)

        errors: List[Tensor] = []
        n_pairs = 0
        n_zero_overlap = 0
        for i in range(K):
            for j in range(i + 1, K):
                n_pairs += 1
                pair_has_overlap = False
                for src, tgt in ((i, j), (j, i)):
                    dist = super()._sample_warp_errors(
                        src_flat_feats=flat_feats[src],
                        src_idx_flat=idx_flat[src],
                        src_valid_lin=valid_lin[src],
                        tgt_feats=feats[tgt],
                        tgt_depth=depth_maps[tgt],
                        tgt_valid=valid_maps[tgt],
                        points_world=points_world,
                        tgt_cam=cameras[tgt],
                        image_hw=(H, W),
                        max_samples=per_dir,
                        depth_tol=self.depth_consistency_tol,
                        oversample_factor=self.oversample_factor,
                    )
                    if dist.numel() == 0:
                        continue
                    pair_has_overlap = True
                    values = dist.detach()
                    if sample_on_cpu:
                        values = values.to("cpu")
                    errors.append(values)
                if not pair_has_overlap:
                    n_zero_overlap += 1

        self._last_overlap_stats = {"n_pairs": n_pairs, "n_zero_overlap": n_zero_overlap}
        if not errors:
            return (torch.tensor(float("nan"), device=device),)

        delta = torch.cat(errors)
        if max_total_samples is not None and delta.numel() > max_total_samples:
            idx = torch.randperm(delta.numel(), device=delta.device)[:max_total_samples]
            delta = delta[idx]

        ed = energy_distance(delta)
        return (ed.to(device),)


class MEt3R_Fast3R_PointConsistency_Energy(MEt3R_Fast3R_PointConsistency):
    """
    Energy distance variant of Fast3R point-consistency metric.
    """
    def __init__(
        self,
        *args,
        max_total_samples: Optional[int] = 4096,
        sample_on_cpu: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.max_total_samples = max_total_samples
        self.sample_on_cpu = sample_on_cpu

    @torch.no_grad()
    def forward(self, images: Tensor, **_kwargs) -> Tuple[Tensor, ...]:
        device = images.device

        delta = self._point_dispersions(images)
        if delta.numel() == 0:
            return (torch.tensor(float("nan"), device=device),)

        if self.sample_on_cpu:
            delta = delta.to("cpu")
        if self.max_total_samples is not None and delta.numel() > self.max_total_samples:
            idx = torch.randperm(delta.numel(), device=delta.device)[:self.max_total_samples]
            delta = delta[idx]

        ed = energy_distance(delta)
        return (ed.to(device),)


class MEt3R_Fast3R_IMQ(MEt3R_Fast3R):
    """
    IMQ kernel MMD variant of Fast3R-driven feature consistency metric.
    
    Uses the inverse multiquadric kernel K(x,y) = (c² + |x-y|²)^{-1/2}
    which is bandwidth-free and more robust to outliers than RBF.
    """

    def __init__(
        self,
        *args,
        c: float = 1.0,
        per_pair_max_samples: Optional[int] = 128,
        max_total_samples: Optional[int] = 4096,
        sample_on_cpu: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.c = c
        self.per_pair_max_samples = per_pair_max_samples
        self.max_total_samples = max_total_samples
        self.sample_on_cpu = sample_on_cpu

    @torch.no_grad()
    def forward(self, images: Tensor, **kwargs) -> Tuple[Tensor, ...]:
        device = images.device
        images = images.to(self._device, dtype=torch.float32)

        per_pair_max_samples = kwargs.pop("per_pair_max_samples", self.per_pair_max_samples)
        max_total_samples = kwargs.pop("max_total_samples", self.max_total_samples)
        sample_on_cpu = kwargs.pop("sample_on_cpu", self.sample_on_cpu)

        if images.ndim != 5 or images.shape[0] != 1:
            raise ValueError("Expected images with shape (1,K,C,H,W)")

        (points_world, cameras, feats, idx_maps, depth_maps, valid_maps, valid_lin) = super()._prepare_warp_data(images)
        K, C, H, W = feats.shape
        flat_feats = feats.permute(0, 2, 3, 1).reshape(K, H * W, C)
        idx_flat = idx_maps.view(K, -1)
        per_dir = None if per_pair_max_samples is None else max(1, per_pair_max_samples // 2)

        errors: List[Tensor] = []
        n_pairs = 0
        n_zero_overlap = 0
        for i in range(K):
            for j in range(i + 1, K):
                n_pairs += 1
                pair_has_overlap = False
                for src, tgt in ((i, j), (j, i)):
                    dist = super()._sample_warp_errors(
                        src_flat_feats=flat_feats[src],
                        src_idx_flat=idx_flat[src],
                        src_valid_lin=valid_lin[src],
                        tgt_feats=feats[tgt],
                        tgt_depth=depth_maps[tgt],
                        tgt_valid=valid_maps[tgt],
                        points_world=points_world,
                        tgt_cam=cameras[tgt],
                        image_hw=(H, W),
                        max_samples=per_dir,
                        depth_tol=self.depth_consistency_tol,
                        oversample_factor=self.oversample_factor,
                    )
                    if dist.numel() == 0:
                        continue
                    pair_has_overlap = True
                    values = dist.detach()
                    if sample_on_cpu:
                        values = values.to("cpu")
                    errors.append(values)
                if not pair_has_overlap:
                    n_zero_overlap += 1

        self._last_overlap_stats = {"n_pairs": n_pairs, "n_zero_overlap": n_zero_overlap}
        if not errors:
            return (torch.tensor(float("nan"), device=device),)

        delta = torch.cat(errors)
        if max_total_samples is not None and delta.numel() > max_total_samples:
            idx = torch.randperm(delta.numel(), device=delta.device)[:max_total_samples]
            delta = delta[idx]

        imq_mmd = mmd_imq(delta, c=self.c)
        return (imq_mmd.to(device),)


class MEt3R_Fast3R_PointConsistency_IMQ(MEt3R_Fast3R_PointConsistency):
    """
    IMQ kernel MMD variant of Fast3R point-consistency metric.
    """
    def __init__(
        self,
        *args,
        c: float = 1.0,
        max_total_samples: Optional[int] = 4096,
        sample_on_cpu: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.c = c
        self.max_total_samples = max_total_samples
        self.sample_on_cpu = sample_on_cpu

    @torch.no_grad()
    def forward(self, images: Tensor, **_kwargs) -> Tuple[Tensor, ...]:
        device = images.device

        delta = self._point_dispersions(images)
        if delta.numel() == 0:
            return (torch.tensor(float("nan"), device=device),)

        if self.sample_on_cpu:
            delta = delta.to("cpu")
        if self.max_total_samples is not None and delta.numel() > self.max_total_samples:
            idx = torch.randperm(delta.numel(), device=delta.device)[:self.max_total_samples]
            delta = delta[idx]

        imq_mmd = mmd_imq(delta, c=self.c)
        return (imq_mmd.to(device),)
