import math
import os
from typing import Optional, Literal, Tuple, List

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, Identity
from einops import rearrange, repeat

# PyTorch3D
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    PerspectiveCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
)

# FeatUp / features
from lpips import LPIPS

# VGGT
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from met3r.met3r import mmd_rbf, DEFAULT_REFERENCE_SIGMA, REFERENCE_SIGMA_BY_K, get_reference_sigma, compute_reference_sigma, energy_distance, mmd_imq


# ------------------------------
# Utilities borrowed from your MEt3R base
# ------------------------------

def freeze_model(m: Module) -> None:
    for p in m.parameters():
        p.requires_grad = False
    m.eval()


def convert_to_buffer(module: torch.nn.Module, persistent: bool = True):
    for _, child in list(module.named_children()):
        convert_to_buffer(child, persistent)
    for name, parameter_or_buffer in (
        *module.named_parameters(recurse=False),
        *module.named_buffers(recurse=False),
    ):
        value = parameter_or_buffer.detach().clone()
        delattr(module, name)
        module.register_buffer(name, value, persistent=persistent)


def _minmax_normalize(scores: Tensor, eps: float = 1e-6, min_spread: float = 1e-3) -> Tensor:
    if scores.ndim != 1:
        raise ValueError(f"Expected 1D scores, got shape {tuple(scores.shape)}")
    if scores.numel() == 0:
        return scores
    min_val = scores.min()
    max_val = scores.max()
    spread = max_val - min_val
    if spread < min_spread:
        return torch.ones_like(scores)
    return (scores - min_val) / (spread + eps)


def _combine_robust_view_scores(
    attention_scores: Tensor,
    cosine_scores: Tensor,
    *,
    attention_weight: float,
    cosine_weight: float,
    eps: float = 1e-6,
) -> Tensor:
    if attention_scores.shape != cosine_scores.shape:
        raise ValueError(
            f"Expected matching score shapes, got {tuple(attention_scores.shape)} and {tuple(cosine_scores.shape)}"
        )
    attention_term = _minmax_normalize(attention_scores.to(torch.float32), eps=eps)
    cosine_term = _minmax_normalize(cosine_scores.to(torch.float32), eps=eps)
    return attention_weight * attention_term + cosine_weight * cosine_term


def _select_surviving_views(view_scores: Tensor, *, threshold: float, anchor_index: int = 0) -> Tensor:
    if view_scores.ndim != 1:
        raise ValueError(f"Expected 1D view scores, got shape {tuple(view_scores.shape)}")
    if not 0 <= anchor_index < view_scores.numel():
        raise ValueError(f"anchor_index={anchor_index} is out of bounds for {view_scores.numel()} scores")
    keep_mask = view_scores >= threshold
    keep_mask[anchor_index] = True
    return keep_mask


def _apply_rejection_penalty(base_score: Tensor, *, rejected_fraction: float, penalty_weight: float) -> Tensor:
    penalty = base_score.new_tensor([penalty_weight * rejected_fraction], dtype=torch.float32)
    return base_score.to(torch.float32) + penalty


class MEt3R_VGGT(Module):
    """
    MEt3R-style consistency metric using VGGT.

    Differences vs. MASt3R/DUSt3R version:
      - Processes K>=2 views *in one forward* through VGGT (no pairwise backbone calls).
      - Builds a single fused point cloud by concatenating per-view world points (optionally confidence-thresholded).
      - Renders that cloud into each predicted camera and measures feature consistency across all pairs.
    """

    def __init__(
        self,
        img_size: Optional[int] = None,
        distance: Literal["cosine", "mse", "rmse", "psnr", "lpips"] = "cosine",
        feature_backbone: Optional[Literal["dino16", "dinov2", "maskclip", "vit", "clip", "resnet50"]] = "dino16",
        feature_backbone_weights: Optional[str] = "mhamilton723/FeatUp",
        upsampler: Optional[Literal["featup", "nearest", "bilinear", "bicubic"]] = "featup",
        use_norm: Optional[bool] = True,
        confidence_threshold: float = 0.0,
        freeze: bool = True,
        rasterizer_kwargs: dict | None = None,
        vggt_weights: str = "facebook/VGGT-1B",
        depth_consistency_tol: float = 0.5,
        samples_per_pair: int = 2048,
        oversample_factor: int = 4,
        feature_batch_size: Optional[int] = None,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.img_size = img_size
        self.distance = distance
        self.upsampler = upsampler
        self.confidence_threshold = confidence_threshold
        self.depth_consistency_tol = depth_consistency_tol
        self.samples_per_pair = samples_per_pair
        self.oversample_factor = oversample_factor
        self.feature_batch_size = feature_batch_size

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        if dtype is None:
            # bfloat16 is supported on Ampere+
            dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
        self._dtype = dtype

        # ---------- Feature extractor (FeatUp or simple backbone) ----------
        if distance == "cosine":
            if "FeatUp" in (feature_backbone_weights or ""):
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
                self.feature_model = torch.hub.load(feature_backbone_weights, feature_backbone)

            if freeze:
                freeze_model(self.feature_model)
                convert_to_buffer(self.feature_model, persistent=False)

        if distance == "lpips":
            self.lpips = LPIPS(spatial=True)

        # ---------- VGGT backbone ----------
        self.vggt = VGGT.from_pretrained(vggt_weights).to(self._device)
        if freeze:
            freeze_model(self.vggt)
            convert_to_buffer(self.vggt, persistent=False)

        # ---------- Rasterizer / compositor ----------
        self.compositor = AlphaCompositor()
        rasterizer_kwargs = rasterizer_kwargs or {}
        # If img_size is None we will reset per-call from input size.
        if self.img_size is not None:
            self.set_rasterizer(image_size=self.img_size, **rasterizer_kwargs)

    # ------------------------------
    # Rendering helpers
    # ------------------------------
    def set_rasterizer(self, image_size, points_per_pixel=10, bin_size=0, radius: float | None = None, **kwargs):
        if radius is None:
            # Default radius suitable for dense point clouds
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

    def render(self, point_clouds: Pointclouds, cameras: PerspectiveCameras, background_value: float = -1e4):
        # Following the parent logic: call rasterizer, then composite manually to recover a validity mask
        with torch.autocast("cuda", enabled=False):
            fragments = self.rasterizer(point_clouds, cameras=cameras)
        r = self.rasterizer.raster_settings.radius
        dists2 = fragments.dists.permute(0, 3, 1, 2)
        weights = 1 - dists2 / (r * r)
        images = self.compositor(
            fragments.idx.long().permute(0, 3, 1, 2),
            weights,
            point_clouds.features_packed().permute(1, 0),
        )
        images = images.permute(0, 2, 3, 1)  # (B,H,W,C)
        zbuf = fragments.zbuf
        # Validity mask: where at least one point hit (idx != -1) across the top-k per-pixel list
        valid = (fragments.idx[..., 0] != -1)  # (B,H,W)
        # Zero out invalid pixels to avoid spurious distances
        images = images * valid.unsqueeze(-1).to(images.dtype)
        return images, valid, zbuf

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

    def _project_points(self, cameras: PerspectiveCameras, points_world: Tensor, eps: float = 1e-6) -> Tuple[Tensor, Tensor]:
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

        tgt_feats_in = tgt_feats.unsqueeze(0)
        tgt_s = F.grid_sample(
            tgt_feats_in,
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

    def _prepare_warp_data(
        self,
        images: Tensor,
    ) -> Tuple[Tensor, List[PerspectiveCameras], Tensor, Tensor, Tensor, Tensor, List[Tensor]]:
        device = self._device
        dtype = self._dtype
        images = images.to(device)
        B, K, _, H, W = images.shape
        if B != 1:
            raise ValueError("MEt3R_VGGT currently supports batch size 1; process samples sequentially.")

        if self.img_size is None:
            self.set_rasterizer(image_size=(H, W))

        with torch.cuda.amp.autocast(enabled=(device.startswith("cuda")), dtype=dtype):
            preds = self.vggt(images)
        world_pts = preds.get("world_points")
        world_conf = preds.get("world_points_conf")
        pose_enc = preds["pose_enc"]

        if world_pts is None:
            raise RuntimeError("VGGT did not return 'world_points'. Ensure you're using a weights variant that predicts 3D.")

        if world_conf is not None:
            self._last_confidence_values = rearrange(world_conf, "B K H W -> (B K H W)").detach().to(
                "cpu", dtype=torch.float32
            )
        else:
            self._last_confidence_values = torch.empty(0, dtype=torch.float32)

        view_ids = torch.arange(K, device=device).view(1, K, 1, 1).expand(B, K, H, W).reshape(-1)
        points_world = rearrange(world_pts, "B K H W C3 -> (B K H W) C3")
        if world_conf is not None and self.confidence_threshold > 0.0:
            conf = rearrange(world_conf, "B K H W -> (B K H W)")
            keep = conf >= self.confidence_threshold
            points_world = points_world[keep]
            view_ids = view_ids[keep]
        self._last_point_view_ids = view_ids

        extri, intri = pose_encoding_to_extri_intri(pose_enc, image_size_hw=(H, W))
        R = extri[0, :, :3, :3]
        t = extri[0, :, :3, 3]
        fx = intri[0, :, 0, 0]
        fy = intri[0, :, 1, 1]
        cx = intri[0, :, 0, 2]
        cy = intri[0, :, 1, 2]

        cams: List[PerspectiveCameras] = []
        for k in range(K):
            cams.append(
                PerspectiveCameras(
                    device=device,
                    R=R[k].unsqueeze(0),
                    T=t[k].unsqueeze(0),
                    focal_length=torch.stack([fx[k], fy[k]], dim=0).view(1, 2),
                    principal_point=torch.stack([cx[k], cy[k]], dim=0).view(1, 2),
                    image_size=((H, W),),
                    in_ndc=False,
                )
            )

        cloud = Pointclouds(points=[points_world])

        idx_maps: List[Tensor] = []
        depth_maps: List[Tensor] = []
        valid_maps: List[Tensor] = []
        valid_lin: List[Tensor] = []
        for cam in cams:
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

        feats = self._extract_dense_features(images[0])
        if self.distance == "cosine":
            self._last_feature_norm_values = torch.linalg.norm(
                feats.detach().to(torch.float32),
                dim=1,
            ).reshape(-1).to("cpu")
        else:
            self._last_feature_norm_values = torch.empty(0, dtype=torch.float32)

        return points_world, cams, feats, idx_tensor, depth_tensor, valid_tensor, valid_lin

    # ------------------------------
    # Features / distances
    # ------------------------------
    def _interpolate(self, lowres_feat: Tensor, ref_images: Tensor) -> Tensor:
        # lowres_feat: (B*K, C, h, w) ; up to (B*K, C, H, W)
        return (
            self.upsampler_model(lowres_feat, ref_images)  # FeatUp path
            if self.upsampler == "featup"
            else F.interpolate(lowres_feat, size=ref_images.shape[-2:], mode=self.upsampler)
        )

    def _get_features(self, images_01: Tensor) -> Tensor:
        return self.feature_model(self.norm(images_01))

    def _extract_dense_features(self, images_01: Tensor) -> Tensor:
        if self.distance != "cosine":
            return images_01

        batch_size = self.feature_batch_size
        if batch_size is None or images_01.shape[0] <= batch_size:
            lowres_feat = self._get_features(images_01)
            dense_feat = self._interpolate(lowres_feat, images_01)
            if dense_feat.shape[-2:] != images_01.shape[-2:]:
                dense_feat = F.interpolate(
                    dense_feat,
                    size=images_01.shape[-2:],
                    mode="bilinear",
                    align_corners=True,
                )
            return dense_feat

        feat_chunks: List[Tensor] = []
        for start in range(0, images_01.shape[0], batch_size):
            image_chunk = images_01[start : start + batch_size]
            lowres_feat = self._get_features(image_chunk)
            dense_feat = self._interpolate(lowres_feat, image_chunk)
            if dense_feat.shape[-2:] != image_chunk.shape[-2:]:
                dense_feat = F.interpolate(
                    dense_feat,
                    size=image_chunk.shape[-2:],
                    mode="bilinear",
                    align_corners=True,
                )
            feat_chunks.append(dense_feat)
        return torch.cat(feat_chunks, dim=0)

    def _distance(self, a: Tensor, b: Tensor, mask: Optional[Tensor] = None, eps: float = 1e-6) -> Tuple[Tensor, Tensor]:
        """Return (score_map[B,1,H,W], weighted_mean[B])"""
        if self.distance == "cosine":
            # a,b: (B,C,H,W)
            num = (a * b).sum(1)
            den = torch.linalg.norm(a, dim=1) * torch.linalg.norm(b, dim=1) + eps
            score_map = 1 - (num / den).clamp(min=-1.0, max=1.0)
            score_map = score_map[:, None]
        elif self.distance == "mse":
            score_map = ((a - b) ** 2).mean(1, keepdim=True)
        elif self.distance == "rmse":
            score_map = ((a - b) ** 2).mean(1, keepdim=True).sqrt()
        elif self.distance == "psnr":
            score_map = 20 * torch.log10(1.0 / (F.mse_loss(a, b, reduction="none").mean(1, keepdim=True).sqrt() + eps))
        elif self.distance == "lpips":
            score_map = self.lpips(2 * a - 1, 2 * b - 1)[:, None]
        else:
            raise NotImplementedError(self.distance)

        if mask is None:
            weighted = score_map.mean(dim=(2, 3))  # (B,1)
        else:
            weighted = (score_map * mask[:, None]).sum(dim=(2, 3)) / (mask.sum(dim=(1, 2))[:, None] + eps)
        return score_map[:, 0], weighted[:, 0]

    # ------------------------------
    # Forward
    # ------------------------------
    @torch.no_grad()
    def forward(
        self,
        images: Tensor,  # (B,K,C,H,W), expected in [0, 1]
        return_overlap_mask: bool = False,
        return_score_map: bool = False,
        return_projections: bool = False,
    ):
        B, K, _, H, W = images.shape
        if K < 2:
            return (torch.full((B,), float("nan"), device=self._device),)

        (
            points_world,
            cams,
            feats,
            idx_maps,
            depth_maps,
            valid_maps,
            valid_lin,
        ) = self._prepare_warp_data(images)

        scores: List[Tensor] = []
        pair_details: List[dict] = []
        n_pairs = 0
        n_zero_overlap = 0
        per_dir = max(1, self.samples_per_pair // 2)
        flat_feats = feats.permute(0, 2, 3, 1).reshape(K, H * W, feats.shape[1])
        idx_flat = idx_maps.view(K, -1)

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
                        tgt_cam=cams[tgt],
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


class MEt3R_VGGT_Robust(MEt3R_VGGT):
    """RobustVGGT-inspired baseline with late-layer view rejection and an explicit reject penalty."""

    def __init__(
        self,
        *args,
        robust_layer_idx: int = 23,
        attention_weight: float = 0.5,
        cosine_weight: float = 0.5,
        rejection_threshold: float = 0.4,
        reject_penalty_weight: float = 0.5,
        anchor_index: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.robust_layer_idx = robust_layer_idx
        self.attention_weight = attention_weight
        self.cosine_weight = cosine_weight
        self.rejection_threshold = rejection_threshold
        self.reject_penalty_weight = reject_penalty_weight
        self.anchor_index = anchor_index

        num_global_blocks = len(self.vggt.aggregator.global_blocks)
        if not 0 <= self.robust_layer_idx < num_global_blocks:
            raise ValueError(
                f"robust_layer_idx={self.robust_layer_idx} is out of bounds for {num_global_blocks} VGGT global blocks"
            )

    def _late_layer_feature_scores(self, layer_tokens: Tensor, patch_start_idx: int) -> Tensor:
        if layer_tokens.ndim != 4 or layer_tokens.shape[0] != 1:
            raise ValueError(f"Expected layer tokens with shape (1,K,T,C), got {tuple(layer_tokens.shape)}")
        patch_tokens = layer_tokens[..., patch_start_idx:, layer_tokens.shape[-1] // 2 :]
        features = F.normalize(patch_tokens[0].to(torch.float32), p=2, dim=-1)
        ref_features = features[self.anchor_index]
        return torch.matmul(features, ref_features.transpose(0, 1)).mean(dim=(1, 2))

    def _late_layer_attention_scores(
        self,
        q_tokens: Tensor,
        k_tokens: Tensor,
        *,
        patch_start_idx: int,
        image_hw: Tuple[int, int],
        num_views: int,
    ) -> Tensor:
        if q_tokens.ndim != 4 or k_tokens.ndim != 4:
            raise ValueError(
                f"Expected q/k tensors with shape (B,H,T,D), got {tuple(q_tokens.shape)} and {tuple(k_tokens.shape)}"
            )

        patch_size = int(self.vggt.aggregator.patch_size)
        image_h, image_w = image_hw
        num_patch_tokens = (image_h // patch_size) * (image_w // patch_size)
        tokens_per_image = patch_start_idx + num_patch_tokens
        query_start = self.anchor_index * tokens_per_image + patch_start_idx
        query_end = query_start + num_patch_tokens
        if query_end > q_tokens.shape[-2]:
            raise ValueError("Anchor patch token range exceeds captured query tokens.")

        total_tokens = num_views * tokens_per_image
        if total_tokens > k_tokens.shape[-2]:
            raise ValueError("Captured key tokens do not cover every input view.")

        q_anchor = q_tokens[:, :, query_start:query_end, :].to(torch.float32)
        k_all = k_tokens[:, :, :total_tokens, :].to(torch.float32)
        scale = 1.0 / math.sqrt(float(q_anchor.shape[-1]))
        attn_logits = torch.einsum("bhqd,bhtd->bhqt", q_anchor, k_all) * scale
        attn_probs = torch.softmax(attn_logits, dim=-1).mean(dim=1).mean(dim=1)[0]

        per_view_scores: List[Tensor] = []
        for view_idx in range(num_views):
            start = view_idx * tokens_per_image + patch_start_idx
            end = start + num_patch_tokens
            per_view_scores.append(attn_probs[start:end].mean())
        return torch.stack(per_view_scores)

    def _compute_view_selection(self, images: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        if images.ndim != 5 or images.shape[0] != 1:
            raise ValueError(f"Expected images with shape (1,K,C,H,W), got {tuple(images.shape)}")
        if not 0 <= self.anchor_index < images.shape[1]:
            raise ValueError(f"anchor_index={self.anchor_index} is out of bounds for K={images.shape[1]}")

        q_cache: List[Tensor] = []
        k_cache: List[Tensor] = []

        def _store_q(_module: Module, _inputs: Tuple[Tensor, ...], output: Tensor) -> None:
            q_cache.append(output.detach())

        def _store_k(_module: Module, _inputs: Tuple[Tensor, ...], output: Tensor) -> None:
            k_cache.append(output.detach())

        attn = self.vggt.aggregator.global_blocks[self.robust_layer_idx].attn
        handles = [
            attn.q_norm.register_forward_hook(_store_q),
            attn.k_norm.register_forward_hook(_store_k),
        ]
        try:
            with torch.cuda.amp.autocast(enabled=self._device.startswith("cuda"), dtype=self._dtype):
                aggregated_tokens_list, patch_start_idx = self.vggt.aggregator(images.to(self._device))
        finally:
            for handle in handles:
                handle.remove()

        if not q_cache or not k_cache:
            raise RuntimeError("Failed to capture late-layer VGGT q/k tensors for robust view scoring.")

        layer_tokens = aggregated_tokens_list[self.robust_layer_idx]
        cosine_scores = self._late_layer_feature_scores(layer_tokens, patch_start_idx)
        attention_scores = self._late_layer_attention_scores(
            q_cache[0],
            k_cache[0],
            patch_start_idx=patch_start_idx,
            image_hw=(images.shape[-2], images.shape[-1]),
            num_views=images.shape[1],
        )
        combined_scores = _combine_robust_view_scores(
            attention_scores,
            cosine_scores,
            attention_weight=self.attention_weight,
            cosine_weight=self.cosine_weight,
        )
        keep_mask = _select_surviving_views(
            combined_scores,
            threshold=self.rejection_threshold,
            anchor_index=self.anchor_index,
        )
        return keep_mask, combined_scores, attention_scores, cosine_scores

    @torch.no_grad()
    def forward(
        self,
        images: Tensor,
        return_overlap_mask: bool = False,
        return_score_map: bool = False,
        return_projections: bool = False,
    ):
        if images.ndim != 5 or images.shape[0] != 1:
            raise ValueError(f"Expected images with shape (1,K,C,H,W), got {tuple(images.shape)}")
        images = images.to(self._device)

        num_views = images.shape[1]
        if num_views < 2:
            return (torch.full((1,), float("nan"), device=self._device),)

        keep_mask, combined_scores, attention_scores, cosine_scores = self._compute_view_selection(images)
        kept_indices = torch.nonzero(keep_mask, as_tuple=False).squeeze(1)
        rejected_views = int((~keep_mask).sum().item())
        rejected_fraction = rejected_views / float(num_views)

        self._last_view_selection = {
            "combined_scores": combined_scores.detach().cpu(),
            "attention_scores": attention_scores.detach().cpu(),
            "cosine_scores": cosine_scores.detach().cpu(),
            "kept_indices": kept_indices.detach().cpu(),
        }

        if kept_indices.numel() < 2:
            score = torch.full((1,), float("nan"), device=self._device)
            self._last_overlap_stats = {
                "n_pairs": 0,
                "n_zero_overlap": 0,
                "n_retained_views": int(kept_indices.numel()),
                "n_rejected_views": rejected_views,
                "rejected_view_fraction": rejected_fraction,
            }
            outputs: List[Tensor | None] = [score]
            if return_overlap_mask:
                outputs.append(None)
            if return_score_map:
                outputs.append(None)
            if return_projections:
                outputs.append(None)
            return tuple(outputs)

        retained_images = images[:, kept_indices, ...]
        base_outputs = super().forward(
            retained_images,
            return_overlap_mask=return_overlap_mask,
            return_score_map=return_score_map,
            return_projections=return_projections,
        )
        outputs = list(base_outputs if isinstance(base_outputs, (tuple, list)) else [base_outputs])
        outputs[0] = _apply_rejection_penalty(
            outputs[0],
            rejected_fraction=rejected_fraction,
            penalty_weight=self.reject_penalty_weight,
        )

        base_stats = dict(getattr(self, "_last_overlap_stats", {}))
        base_stats.update(
            {
                "n_retained_views": int(kept_indices.numel()),
                "n_rejected_views": rejected_views,
                "rejected_view_fraction": rejected_fraction,
            }
        )
        self._last_overlap_stats = base_stats
        return tuple(outputs)


class MEt3R_VGGT_MMD(MEt3R_VGGT):
    """
    Variant of the VGGT-based metric that aggregates per-pixel errors into a
    1-sample MMD² score against an ideal zero-error distribution, mirroring
    the behaviour of MEt3R_MMD.
    
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
        Compute an MMD² score over pooled per-pixel errors across all pairs.

        If gt_images is provided, we additionally compute a two-sample MMD between
        the error distributions of `images` and `gt_images`, mirroring MEt3R_MMD.
        
        Args:
            reference_sigma: Override instance reference_sigma for this call.
            use_reference_sigma: Override instance use_reference_sigma for this call.
            use_per_k_sigma: Override instance use_per_k_sigma for this call.
        """
        if images.ndim != 5:
            raise ValueError(f"Expected images with shape (B,K,C,H,W), got {tuple(images.shape)}")

        device = images.device
        b, k, c, h, w = images.shape

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
                raise ValueError("MEt3R_VGGT_MMD currently supports batch size 1; process samples sequentially.")

            (
                points_world,
                cams,
                feats,
                idx_maps,
                depth_maps,
                valid_maps,
                valid_lin,
            ) = super(MEt3R_VGGT_MMD, self)._prepare_warp_data(batch)

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
                        dist = super(MEt3R_VGGT_MMD, self)._sample_warp_errors(
                            src_flat_feats=flat_feats[src],
                            src_idx_flat=idx_flat[src],
                            src_valid_lin=valid_lin[src],
                            tgt_feats=feats[tgt],
                            tgt_depth=depth_maps[tgt],
                            tgt_valid=valid_maps[tgt],
                            points_world=points_world,
                            tgt_cam=cams[tgt],
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
            gt_batch = gt_images.to(device)
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


class MEt3R_VGGT_PointConsistency(MEt3R_VGGT):
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
            raise ValueError("MEt3R_VGGT_PointConsistency currently supports batch size 1; process samples sequentially.")

        (
            points_world,
            cams,
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

                uv, z = self._project_points(cams[k], pts_chunk, eps=eps)
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


class MEt3R_VGGT_PointConsistency_MMD(MEt3R_VGGT_PointConsistency):
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


class MEt3R_VGGT_Energy(MEt3R_VGGT):
    """
    Energy distance variant of VGGT-driven feature consistency metric.
    
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
        if images.ndim != 5 or images.shape[0] != 1:
            raise ValueError("Expected images with shape (1,K,C,H,W)")
        device = images.device

        per_pair_max_samples = kwargs.pop("per_pair_max_samples", self.per_pair_max_samples)
        max_total_samples = kwargs.pop("max_total_samples", self.max_total_samples)
        sample_on_cpu = kwargs.pop("sample_on_cpu", self.sample_on_cpu)

        (points_world, cams, feats, idx_maps, depth_maps, valid_maps, valid_lin) = super()._prepare_warp_data(images)
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
                        tgt_cam=cams[tgt],
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


class MEt3R_VGGT_PointConsistency_Energy(MEt3R_VGGT_PointConsistency):
    """
    Energy distance variant of VGGT point-consistency metric.
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


class MEt3R_VGGT_IMQ(MEt3R_VGGT):
    """
    IMQ kernel MMD variant of VGGT-driven feature consistency metric.
    
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
        if images.ndim != 5 or images.shape[0] != 1:
            raise ValueError("Expected images with shape (1,K,C,H,W)")
        device = images.device

        per_pair_max_samples = kwargs.pop("per_pair_max_samples", self.per_pair_max_samples)
        max_total_samples = kwargs.pop("max_total_samples", self.max_total_samples)
        sample_on_cpu = kwargs.pop("sample_on_cpu", self.sample_on_cpu)

        (points_world, cams, feats, idx_maps, depth_maps, valid_maps, valid_lin) = super()._prepare_warp_data(images)
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
                        tgt_cam=cams[tgt],
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


class MEt3R_VGGT_PointConsistency_IMQ(MEt3R_VGGT_PointConsistency):
    """
    IMQ kernel MMD variant of VGGT point-consistency metric.
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
