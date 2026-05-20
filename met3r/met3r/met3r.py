

import sys
import os
import os.path as path

from typing import Literal, Optional, Union

import torch

from torch import Tensor
from torch.nn import Identity, functional as F
from torch.nn import Module
from pathlib import Path
from jaxtyping import Float, Bool
from typing import Union, Tuple
from einops import rearrange, repeat
from torchvision.models.optical_flow import raft_large
from torchmetrics.functional.image import structural_similarity_index_measure


# Load Pytorch3D
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    FoVPerspectiveCameras, 
    PerspectiveCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
)


from lpips import LPIPS


def _find_mast3r_repo_path() -> str:
    candidates: list[Path] = []
    env_path = os.environ.get("MAST3R_REPO_PATH")
    if env_path:
        candidates.append(Path(env_path).expanduser())

    cwd = Path.cwd().resolve()
    repo_root = Path(__file__).resolve().parents[2]
    candidates.extend(
        [
            cwd.parent / "mast3r",
            cwd / "mast3r",
            repo_root / "mast3r",
            repo_root.parent / "mast3r",
        ]
    )

    for candidate in candidates:
        if (candidate / "mast3r").is_dir() and (candidate / "dust3r" / "dust3r").is_dir():
            return str(candidate.resolve())

    checked = ", ".join(str(candidate) for candidate in candidates)
    raise ImportError(f"mast3r and dust3r are not initialized; checked: {checked}")


MASt3R_REPO_PATH = _find_mast3r_repo_path()
DUSt3R_REPO_PATH = path.join(MASt3R_REPO_PATH, 'dust3r')
MASt3R_LIB_PATH = path.join(MASt3R_REPO_PATH, 'mast3r')
DUSt3R_LIB_PATH = path.join(DUSt3R_REPO_PATH, 'dust3r')
sys.path.insert(0, MASt3R_REPO_PATH)
sys.path.insert(0, DUSt3R_REPO_PATH)
from dust3r.utils.geometry import xy_grid

def freeze_model(m: Module) -> None:
    for param in m.parameters():
        param.requires_grad = False
    m.eval()

def convert_to_buffer(module: torch.nn.Module, persistent: bool = True):
    # Recurse over child modules.
    for name, child in list(module.named_children()):
        convert_to_buffer(child, persistent)

    # Also re-save buffers to change persistence.
    for name, parameter_or_buffer in (
        *module.named_parameters(recurse=False),
        *module.named_buffers(recurse=False),
    ):
        value = parameter_or_buffer.detach().clone()
        delattr(module, name)
        module.register_buffer(name, value, persistent=persistent)

backbone_to_weights = {
    "mast3r": "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
    "dust3r": "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
}

# Default reference sigma calibrated on consistent MipNeRF360 scenes (cosine distance).
# Use this for fair comparisons across samples with different error distributions.
DEFAULT_REFERENCE_SIGMA: float = 0.15

# Per-K reference sigmas calibrated on consistent MipNeRF360 scenes.
# Keys are view subset sizes (K), values are median sigmas from consistent scenes.
# If K is not in this dict, fall back to DEFAULT_REFERENCE_SIGMA.
REFERENCE_SIGMA_BY_K: dict[int, float] = {
    3: 0.14,
    4: 0.15,
    5: 0.15,
    6: 0.16,
    7: 0.16,
    8: 0.16,
}


def get_reference_sigma(k: int | None = None) -> float:
    """
    Get reference sigma for a given view subset size K.
    
    Args:
        k: Number of views in the subset. If None, returns DEFAULT_REFERENCE_SIGMA.
    
    Returns:
        Reference sigma calibrated for that K, or DEFAULT_REFERENCE_SIGMA if K not found.
    """
    if k is None:
        return DEFAULT_REFERENCE_SIGMA
    return REFERENCE_SIGMA_BY_K.get(k, DEFAULT_REFERENCE_SIGMA)


def compute_reference_sigma(errors: Tensor, n_samples: int = 512, eps: float = 1e-8) -> float:
    """
    Compute a reference kernel bandwidth from a set of errors.
    
    This should be called once on a representative set of errors (e.g., pooled
    errors from all consistent/ground-truth samples) and the result passed to
    mmd_rbf() for all subsequent evaluations.
    
    Args:
        errors: Tensor of error values to compute bandwidth from.
        n_samples: Number of samples to use for median computation (for speed).
        eps: Minimum bandwidth to avoid division by zero.
    
    Returns:
        sigma: The median-heuristic bandwidth suitable for RBF kernels.
    """
    e = errors.flatten()
    n = e.numel()
    if n < 2:
        return eps
    with torch.no_grad():
        idx = torch.randperm(n, device=e.device)[:min(n_samples, n)]
        diff = (e[idx][:, None] - e[idx][None, :]).abs()
        sigma = diff[diff > 0].median().clamp(min=eps)
    return float(sigma.item()) if isinstance(sigma, Tensor) else float(sigma)


def mmd_rbf(
    errors: Tensor,  # shape [N]  - all valid delta values pooled together
    sigma: float | None = None,
    return_sigma: bool = False,
    eps: float = 1e-8,
    use_reference_sigma: bool = False,
):
    """
    Unbiased MMD² between the empirical distribution of errors and a Dirac delta at 0 (ideal consistency).
    
    Args:
        errors: Tensor of shape [N] containing error values.
        sigma: Fixed kernel bandwidth. If None and use_reference_sigma=False, computed adaptively.
        return_sigma: If True, also return the sigma used.
        eps: Minimum bandwidth to avoid numerical issues.
        use_reference_sigma: If True and sigma is None, use DEFAULT_REFERENCE_SIGMA instead
            of computing adaptively. This is recommended for calibration/comparison tasks
            where you want scores to be comparable across different samples.
    
    Returns:
        mmd2: The unbiased MMD² estimate.
        sigma (optional): The bandwidth used, if return_sigma=True.
    
    Note:
        The adaptive sigma (computed from median pairwise distances) can destroy
        discriminative power when comparing samples with different error magnitudes,
        as it normalizes each distribution to a similar scale. For calibration tasks,
        either pass a fixed sigma or set use_reference_sigma=True.
    """
    e = errors.flatten()  # (N,)
    n = e.numel()
    if n < 2:  # trivial edge-case
        out = torch.zeros(1, device=e.device)
        return (out, torch.tensor(float('nan'), device=e.device)) if return_sigma else out

    # bandwidth selection
    if sigma is None:
        if use_reference_sigma:
            sigma = DEFAULT_REFERENCE_SIGMA
        else:
            # adaptive bandwidth: median of pairwise |delta_i-delta_j|
            with torch.no_grad():
                idx = torch.randperm(n, device=e.device)[:min(512, n)]
                diff = (e[idx][:, None] - e[idx][None, :]).abs()
                sigma = diff[diff > 0].median().clamp(min=eps)

    k = torch.exp(-(e[:, None] - e[None, :]) ** 2 / (2 * sigma**2))  # (N, N)

    # unbiased estimate: see Gretton et al. 2012, eq.(6)
    mmd2 = (k.sum() - k.trace()) / (n * (n - 1))  # first term
    mmd2 += -2 * torch.exp(-(e**2) / (2 * sigma**2)).mean()  # second term
    mmd2 += 1  # k(0,0)

    return (mmd2, sigma) if return_sigma else mmd2


def energy_distance(
    errors: Tensor,  # shape [N] - all valid delta values pooled together
) -> Tensor:
    """
    Energy distance between the empirical distribution of errors and a Dirac delta at 0.
    
    This is a kernel-free alternative to MMD that doesn't require bandwidth selection.
    Energy distance = 2*E[|X|] - E[|X-X'|] where X,X' are iid samples from the error distribution
    and the reference is a point mass at 0.
    
    For our case (reference = delta_0):
        E_energy = 2*E[|errors|] - E[|errors_i - errors_j|]
    
    This measures how far the error distribution is from the ideal (all zeros).
    Higher values = worse consistency (errors far from 0).
    
    Args:
        errors: Tensor of shape [N] containing error values (should be non-negative).
    
    Returns:
        energy: The energy distance estimate (scalar tensor).
    
    References:
        Székely & Rizzo (2013). Energy statistics: A class of statistics based on distances.
    """
    e = errors.flatten()  # (N,)
    n = e.numel()
    if n < 2:
        return torch.zeros(1, device=e.device)
    
    # Term 1: 2 * E[|X - 0|] = 2 * E[|X|] = 2 * mean(errors)
    # (errors are already distances/absolute values, so |e| = e)
    term1 = 2.0 * e.mean()
    
    # Term 2: E[|X - X'|] - expected pairwise distance
    # For efficiency, use subsampling if N is large
    if n > 1024:
        idx = torch.randperm(n, device=e.device)[:1024]
        e_sub = e[idx]
        term2 = (e_sub[:, None] - e_sub[None, :]).abs().mean()
    else:
        term2 = (e[:, None] - e[None, :]).abs().mean()
    
    return term1 - term2


def mmd_imq(
    errors: Tensor,
    c: float = 1.0,
) -> Tensor:
    """
    MMD using inverse multiquadric (IMQ) kernel: K(x,y) = (c² + |x-y|²)^{-1/2}.
    
    The IMQ kernel is bandwidth-free (c is a fixed constant) and has heavier tails
    than RBF, making it more robust to outliers. Unlike RBF, it doesn't require
    tuning sigma per dataset.
    
    For measuring distance to Dirac delta at 0:
        MMD² = E[K(X,X')] - 2*E[K(X,0)] + K(0,0)
             = E[(c² + |X-X'|²)^{-1/2}] - 2*E[(c² + |X|²)^{-1/2}] + 1/c
    
    Args:
        errors: Tensor of shape [N] containing error values.
        c: Kernel scale parameter (default=1.0). Larger c = smoother kernel.
    
    Returns:
        mmd: The MMD estimate (scalar tensor).
    """
    e = errors.flatten()
    n = e.numel()
    if n < 2:
        return torch.zeros(1, device=e.device)
    
    c2 = c * c
    
    # Term 1: E[K(X,X')] (unbiased: exclude diagonal), using subsampling for efficiency.
    if n > 1024:
        idx = torch.randperm(n, device=e.device)[:1024]
        e_sub = e[idx]
        m = e_sub.numel()
        if m < 2:
            return torch.zeros(1, device=e.device)
        pairwise_sq = (e_sub[:, None] - e_sub[None, :]) ** 2
        k_xx = (c2 + pairwise_sq).rsqrt()
        term1 = (k_xx.sum() - k_xx.diag().sum()) / (m * (m - 1))
    else:
        pairwise_sq = (e[:, None] - e[None, :]) ** 2
        k_xx = (c2 + pairwise_sq).rsqrt()
        term1 = (k_xx.sum() - k_xx.diag().sum()) / (n * (n - 1))
    
    # Term 2: -2 * E[K(X, 0)] = -2 * E[(c² + |X|²)^{-1/2}]
    term2 = -2.0 * (c2 + e ** 2).rsqrt().mean()
    
    # Term 3: K(0,0) = 1/c
    term3 = 1.0 / c
    
    return term1 + term2 + term3


class MEt3R(Module):

    def __init__(
        self, 
        img_size: Optional[int] = 256, 
        use_norm: Optional[bool]=True,
        backbone: Literal["mast3r", "dust3r", "raft"] = "mast3r",
        feature_backbone: Optional[Literal["dino16", "dinov2", "maskclip", "vit", "clip", "resnet50"]] = "dino16",
        feature_backbone_weights: Optional[Union[str, Path]] = "mhamilton723/FeatUp",
        upsampler: Optional[Literal["featup", "nearest", "bilinear", "bicubic"]] = "featup",
        distance: Literal["cosine", "lpips", "rmse", "psnr", "mse", "ssim"] = "cosine",
        freeze: bool=True,
        rasterizer_kwargs: dict = {}
    ) -> None:
        """Initialize MET3R

        Args:
            img_size (int, optional): Image size for rasterization. Set to None to allow for rasterization with the input resolution on the fly. Defaults to 224.
            use_norm (bool, optional): Whether to use norm layers in FeatUp. Refer to https://github.com/mhamilton723/FeatUp?tab=readme-ov-file#using-pretrained-upsamplers. Defaults to True.
            feature_backbone (str, optional): Feature backbone for FeatUp. Select from ["dino16", "dinov2", "maskclip", "vit", "clip", "resnet50"]. Defaults to "dino16".
            feature_backbone_weights (str | Path, optional): Weight path for FeatUp upsampler. Defaults to "mhamilton723/FeatUp".
            upsampler (str, optional): Set upsampling types. Defaults to "featup".
            distance (str): Select which distance to compute. Default to "cosine" for computing feature dissimilarity.
            freeze (bool, optional): Set whether to freeze the model. Defaults to True.
            rasterizer_kwargs (dict): Additional argument for point cloud render from PyTorch3D. Default to an empty dict. 
        """
        super().__init__()
        if isinstance(feature_backbone_weights, str) and feature_backbone_weights == "mhamilton723/FeatUp":
            feature_backbone_weights = "mhamilton723/FeatUp:main"
        self.img_size = img_size
        self.upsampler = upsampler
        self.backbone = backbone
        self.distance = distance
        if upsampler == "featup" and "FeatUp" not in feature_backbone_weights:
            raise ValueError("Need to specify the correct weight path on huggingface for using `upsampler=\"featup\"`. Set `feature_backbone_weights=\"mhamilton723/FeatUp\"`")
            
        if distance == "cosine":
            if "FeatUp" in feature_backbone_weights:
                # Load featup
                from featup.util import norm, unnorm
                self.norm = norm
                if feature_backbone not in ["dino16", "dinov2", "maskclip", "vit", "clip", "resnet50"]:
                    raise ValueError("Provide `feature_backone` is not implemented for `FeatUp`. Please select from [\"dino16\", \"dinov2\", \"maskclip\", \"vit\", \"clip\", \"resnet50\"] in conjunction with `feature_backbone_weights=\"mhamilton723/FeatUp\"`")
                if use_norm is None:
                    raise ValueError("When using `FeatUp`, specify `use_norm` as either `True` or `False`. Currently it is set to `None`")
                
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
            

        
        
        if backbone == "mast3r":
            from mast3r.model import AsymmetricMASt3R 
            self.backbone_model = AsymmetricMASt3R.from_pretrained(backbone_to_weights[backbone])
        elif backbone == "dust3r":
            from dust3r.model import AsymmetricCroCo3DStereo 
            self.backbone_model = AsymmetricCroCo3DStereo.from_pretrained(backbone_to_weights[backbone])
        elif backbone == "raft":
            self.backbone_model = raft_large(pretrained=True, progress=False)
        else:
            raise NotImplementedError("Specificed backbone for warping is not available. Please select from ['mast3r', 'dust3r', 'raft']")  

        if freeze:
            freeze_model(self.backbone_model) 
            convert_to_buffer(self.backbone_model, persistent=False)

        if backbone in ["mast3r", "dust3r"]:

            if self.img_size is not None:
                self.set_rasterizer(
                    image_size=img_size, 
                    points_per_pixel=10,
                    bin_size=0,
                    **rasterizer_kwargs
                )
            
            self.compositor = AlphaCompositor()
        
        if distance == "lpips":
            self.lpips = LPIPS(spatial=True)

    def _distance(self, inp1: Tensor, inp2: Tensor, mask: Optional[Tensor]=None, eps: float=1e-5):

        if self.distance == "cosine":
            # Get feature dissimilarity score map
            score_map = 1 - (inp1 * inp2).sum(1) / (torch.linalg.norm(inp1, dim=1) * torch.linalg.norm(inp2, dim=1) + eps) 
            score_map = score_map[:, None]
        elif self.distance == "mse":
            score_map = ((inp1 - inp2)**2).mean(1, keepdim=True)
        elif self.distance == "psnr":
            score_map = 20 * torch.log10(255.0 / (torch.sqrt(((inp1 - inp2)**2)).mean(1, keepdim=True) + eps))
        elif self.distance == "rmse":
            score_map = ((inp1 - inp2)**2).mean(1, keepdim=True)**0.5
        elif self.distance == "lpips":
            score_map = self.lpips(2 * inp1 - 1, 2 * inp2 - 1)
            score_map = score_map[:, None]
        elif self.distance == "ssim":
            _, score_map = structural_similarity_index_measure(inp1, inp2, return_full_image=True)
            print(score_map.shape)
            print(mask.shape)
        result = [score_map[:, 0]]
        if mask is not None: 
            # Weighted averate of score map with computed mask
            weighted = (score_map * mask[:, None]).sum(-1).sum(-1)  / (mask[:, None].sum(-1).sum(-1) + eps)
            result.append(weighted.mean(1))

        return tuple(result)
    
    def _interpolate(self, inp1: Tensor, inp2: Tensor):

        if self.upsampler == "featup":
            feat = self.upsampler_model(inp1, inp2)
            # Important for specific backbone which may not return with correct dimensions
            feat = F.interpolate(feat, (inp2.shape[-2:]), mode="bilinear")
        else:

            feat = F.interpolate(inp1, (inp2.shape[-2:]), mode=self.upsampler)

        return feat
    
    def _get_features(self, images):
        
        return self.feature_model(self.norm(images))

    def set_rasterizer(
        self,
        image_size, 
        points_per_pixel=10,
        bin_size=0,
        **kwargs
    ) -> None:
        raster_settings = PointsRasterizationSettings(
            image_size=image_size, 
            points_per_pixel=points_per_pixel,
            bin_size=bin_size,
            **kwargs
        )

        self.rasterizer = PointsRasterizer(cameras=None, raster_settings=raster_settings)

    def render(
        self, 
        point_clouds: Pointclouds, 
        **kwargs
    ) -> Tuple[
            Float[Tensor, "b h w c"], 
            Float[Tensor, "b 2 h w n"]
        ]:
        """Adoped from Pytorch3D https://pytorch3d.readthedocs.io/en/latest/modules/renderer/points/renderer.html

        Args:
            point_clouds (pytorch3d.structures.PointCloud): Point cloud object to render 

        Returns:
            images (Float[Tensor, "b h w c"]): Rendered images
            zbuf (Float[Tensor, "b k h w n"]): Z-buffers for points per pixel
        """
        with torch.autocast("cuda", enabled=False):
            fragments = self.rasterizer(point_clouds, **kwargs)

        r = self.rasterizer.raster_settings.radius

        dists2 = fragments.dists.permute(0, 3, 1, 2)
        weights = 1 - dists2 / (r * r)
        images = self.compositor(
            fragments.idx.long().permute(0, 3, 1, 2),
            weights,
            point_clouds.features_packed().permute(1, 0),
            **kwargs,
        )

        # permute so image comes at the end
        images = images.permute(0, 2, 3, 1)

        return images, fragments.zbuf
    
    def warp_image(self, image: torch.Tensor, flow: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Warp an input image using an optical flow field and compute a mask for gaps.

        Args:
            image (torch.Tensor): The input image of shape (B, C, H, W), where
                                B is the batch size,
                                C is the number of channels,
                                H is the height,
                                W is the width.
            flow (torch.Tensor): The optical flow of shape (B, 2, H, W), where the 2 channels
                                correspond to the horizontal and vertical flow components.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - The warped image of shape (B, C, H, W).
                - A mask of shape (B, 1, H, W) indicating gaps due to warping (1 for valid pixels, 0 for gaps).
        """
        B, C, H, W = image.shape

        # Generate a grid of coordinates for the image
        y, x = torch.meshgrid(
            torch.arange(H, device=image.device, dtype=torch.float32),
            torch.arange(W, device=image.device, dtype=torch.float32),
            indexing="ij"
        )

        # Normalize the grid coordinates to the range [-1, 1]
        x = x / (W - 1) * 2 - 1
        y = y / (H - 1) * 2 - 1

        grid = torch.stack((x, y), dim=2).unsqueeze(0)  # Shape: (1, H, W, 2)
        grid = grid.repeat(B, 1, 1, 1)  # Repeat for batch size

        # Normalize flow from pixel space to normalized coordinates
        flow = flow.clone()
        flow[:, 0, :, :] = flow[:, 0, :, :] / (W - 1) * 2  # Normalize horizontal flow
        flow[:, 1, :, :] = flow[:, 1, :, :] / (H - 1) * 2  # Normalize vertical flow

        # Add the flow to the grid
        flow = flow.permute(0, 2, 3, 1)  # Shape: (B, H, W, 2)
        warped_grid = grid + flow

        # Clip grid values to ensure they are within bounds
        warped_grid[..., 0] = torch.clamp(warped_grid[..., 0], -1, 1)
        warped_grid[..., 1] = torch.clamp(warped_grid[..., 1], -1, 1)

        # Use grid_sample to warp the image
        warped_image = F.grid_sample(image, warped_grid, mode="bilinear", padding_mode="border", align_corners=True)

        # Compute a mask for valid pixels
        mask = F.grid_sample(
            torch.ones((B, 1, H, W), device=image.device, dtype=image.dtype),
            warped_grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        mask = (mask > 0.999).float()  # Threshold to create a binary mask

        return warped_image, mask

    def forward(
        self, 
        images: Float[Tensor, "b 2 c h w"], 
        return_overlap_mask: bool=False, 
        return_score_map: bool=False, 
        return_projections: bool=False
    ) -> Tuple[
            float, 
            Bool[Tensor, "b h w"] | None, 
            Float[Tensor, "b h w"] | None, 
            Float[Tensor, "b 2 c h w"] | None
        ]:
        
        """Forward function to compute MET3R
        Args:
            images (Float[Tensor, "b 2 c h w"]): Normalized input image pairs with values ranging in [-1, 1],
            return_overlap_mask (bool, False): Return 2D map overlapping mask
            return_score_map (bool, False): Return 2D map of feature dissimlarity (Unweighted) 
            return_projections (bool, False): Return projected feature maps

        Return:
            score (Float[Tensor, "b"]): MET3R score which consists of weighted mean of feature dissimlarity
            mask (bool[Tensor, "b c h w"], optional): Overlapping mask
            feat_dissim_maps (bool[Tensor, "b h w"], optional): Feature dissimilarity score map
            proj_feats (bool[Tensor, "b h w c"], optional): Projected and rendered features
        """
        
        *_, h, w = images.shape
        
        # Set rasterization settings on the fly based on input resolution
        if self.img_size is None:
            raster_settings = PointsRasterizationSettings(
                    image_size=(h, w), 
                    radius = 0.01,
                    points_per_pixel = 10,
                    bin_size=0
                )
            self.rasterizer = PointsRasterizer(cameras=None, raster_settings=raster_settings)

        
        b, k, *_ = images.shape
        images = rearrange(images, "b k c h w -> (b k) c h w")
        images = (images + 1) / 2

        if self.distance == "cosine":
            # NOTE: Compute features
            lr_feat = self._get_features(images)
            # NOTE: Transform feature to higher resolution either using `interpolate` or `FeatUp`
            hr_feat = self._interpolate(lr_feat, images)
            # K=2 since we only compare an image pairs
            hr_feat = rearrange(hr_feat, "(b k) ... -> b k ...", k=2)
            self._last_feature_norm_values = torch.linalg.norm(
                hr_feat.detach().to(torch.float32),
                dim=2,
            ).reshape(-1).to("cpu")
        else:
            self._last_feature_norm_values = torch.empty(0, dtype=torch.float32)
        images = rearrange(images, "(b k) ... -> b k ...", k=2)
        images = 2 * images - 1

        # NOTE: Apply Backbone MASt3R/DUSt3R/RAFT to warp one view to the other and compute overlap masks
        if self.backbone == "raft":
            self._last_confidence_values = torch.empty(0, dtype=torch.float32)
            flow = self.backbone_model(images[:, 0, ...], images[:, 1, ...])[-1]

            if self.distance == "cosine":
                view1 = hr_feat[:, 0, ...]
                view2 = hr_feat[:, 1, ...]
            else:
                view1 = images[:, 0, ...]
                view2 = images[:, 1, ...]

            warped_view, mask = self.warp_image(view2, flow)
            rendering = torch.stack([view1, warped_view], dim=1)

        else:
            view1 = {"img": images[:, 0, ...], "instance": [""]}
            view2 = {"img": images[:, 1, ...], "instance": [""]}
            pred1, pred2 = self.backbone_model(view1, view2)

            ptmps = torch.stack([pred1["pts3d"], pred2["pts3d_in_other_view"]], dim=1).detach()
            conf = torch.stack([pred1["conf"], pred2["conf"]], dim=1).detach()
            self._last_confidence_values = conf.reshape(-1).to("cpu", dtype=torch.float32)

            # NOTE: Get canonical point map using the confidences
            confs11 = conf.unsqueeze(-1) - 0.999
            canon = (confs11 * ptmps).sum(1) / confs11.sum(1)
            
            # Define principal point
            pp = torch.tensor([w /2 , h / 2], device=canon.device)
            
            
            # NOTE: Estimating fx and fy for a given canonical point map
            B, H, W, THREE = canon.shape
            assert THREE == 3

            # centered pixel grid
            pixels = xy_grid(W, H, device=canon.device).view(1, -1, 2) - pp.view(-1, 1, 2)  # B,HW,2
            canon = canon.flatten(1, 2)  # (B, HW, 3)

            # direct estimation of focal
            u, v = pixels.unbind(dim=-1)
            x, y, z = canon.unbind(dim=-1)
            fx_votes = (u * z) / x
            fy_votes = (v * z) / y

            # assume square pixels, hence same focal for X and Y
            f_votes = torch.stack((fx_votes.view(B, -1), fy_votes.view(B, -1)), dim=-1)
            focal = torch.nanmedian(f_votes, dim=-2)[0]
            
            # Normalized focal length
            focal[..., 0] = 1 + focal[..., 0]/w
            focal[..., 1] = 1 + focal[..., 1]/h
            focal = repeat(focal, "b c -> (b k) c", k=2)
            # NOTE: Unproject feature on the point cloud
            ptmps = rearrange(ptmps, "b k h w c -> (b k) (h w) c", b=b, k=2)
            if self.distance == "cosine":
                features = rearrange(hr_feat, "b k c h w -> (b k) (h w) c", k=2)

            else:
                images = (images + 1) / 2
                features = rearrange(images, "b k c h w-> (b k) (h w) c", k=2)
            point_cloud = Pointclouds(points=ptmps, features=features)
            
            # NOTE: Project and Render
            R = torch.eye(3)
            R[0, 0] *= -1
            R[1, 1] *= -1
            R = repeat(R, "... -> (b k) ...", b=b, k=2)
            T = torch.zeros((3, ))
            T = repeat(T, "... -> (b k) ...", b=b, k=2)

            # Define Pytorch3D camera for projection
            cameras = PerspectiveCameras(device=ptmps.device, R=R, T=T, focal_length=focal)
            # Render via point rasterizer to get projected features
            with torch.autocast("cuda", enabled=False):
                rendering, zbuf = self.render(point_cloud, cameras=cameras, background_color=[-10000] * features.shape[-1])
            rendering = rearrange(rendering, "(b k) h w c -> b k c h w",  b=b, k=2)
            
            # Compute overlapping mask
            non_overlap_mask = (rendering == -10000)
            overlap_mask = (1 - non_overlap_mask.float()).prod(2).prod(1)
            
            # Zero out regions which do not overlap
            rendering[non_overlap_mask] = 0.0

            # Mask for weighted sum
            mask = overlap_mask

        # NOTE: Uncomment for incorporating occlusion masks along with overlap mask
        # zbuf = rearrange(zbuf, "(b k) ... -> b k ...",  b=b, k=2)
        # closest_z = zbuf[..., 0]
        # diff = (closest_z[:, 0, ...] - closest_z[:, 1, ...]).abs()
        # mask = (~(diff > 0.5) * (closest_z != -1).prod(1)) * mask
        
        # NOTE: Compute scores as either feature dissimilarity, RMSE, LPIPS, SSIM, MSE, or PSNR 
        score_map, weighted = self._distance(rendering[:, 0, ...], rendering[:, 1, ...], mask=mask)

        outputs = [weighted]
        if return_overlap_mask:
            outputs.append(mask)
            
        if return_score_map:
            outputs.append(score_map)
        
        if return_projections:
            outputs.append(rendering)

        return (*outputs, )


class MEt3R_MMD(MEt3R):
    """
    Inherits every helper (warping, projection, distance, …) from your
    original MEt3R but returns an MMD score instead of the weighted mean.
    
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
    def __init__(self,
                 *args,
                 per_pair_max_samples: Optional[int] = 128,
                 max_total_samples: Optional[int] = 4096,
                 pair_details_max_samples: Optional[int] = None,
                 sample_on_cpu: bool = True,
                 reference_sigma: Optional[float] = None,
                 use_reference_sigma: bool = False,
                 use_per_k_sigma: bool = False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.per_pair_max_samples = per_pair_max_samples
        self.max_total_samples = max_total_samples
        self.pair_details_max_samples = pair_details_max_samples
        self.sample_on_cpu = sample_on_cpu
        self.reference_sigma = reference_sigma
        self.use_reference_sigma = use_reference_sigma
        self.use_per_k_sigma = use_per_k_sigma

    @torch.no_grad()
    def forward(self,
                images: Float[Tensor, "b k c h w"],
                gt_images: Float[Tensor, "b k c h w"] | None = None,
                return_sigma: bool = False,
                **kwargs):
        """
        Args
        ----
        images     : predicted frames  (k may be > 2 - we will compare *all* pairs)
        gt_images  : optional ground-truth frames aligned with `images`.
                     If given, we compute two error sets and return
                     MMD(pred delta  ↔ 0)  and  MMD(pred delta ↔ gt delta).
        return_sigma : debug - return the kernel bandwidth chosen.
        reference_sigma : override instance reference_sigma for this call.
        use_reference_sigma : override instance use_reference_sigma for this call.
        use_per_k_sigma : override instance use_per_k_sigma for this call.
        """
        b, k, c, h, w = images.shape
        device = images.device

        per_pair_max_samples = kwargs.pop("per_pair_max_samples", self.per_pair_max_samples)
        max_total_samples = kwargs.pop("max_total_samples", self.max_total_samples)
        pair_details_max_samples = kwargs.pop("pair_details_max_samples", self.pair_details_max_samples)
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

        # -----  build all pairwise dissimilarity maps (reuse parent helpers)
        def _subsample_values(values: Tensor, max_samples: Optional[int], seed: int) -> Tensor:
            if max_samples is None or values.numel() <= max_samples:
                return values
            generator = torch.Generator(device=values.device)
            generator.manual_seed(seed)
            idx = torch.randperm(values.numel(), generator=generator, device=values.device)[:max_samples]
            return values[idx]

        def _collect_errors(
            frames: Float[Tensor, "b k c h w"] | None,
        ) -> tuple[list[Tensor], int, int, list[dict], list[Tensor], list[Tensor]]:
            if frames is None:
                return [], 0, 0, [], [], []
            _, k_local, _, _, _ = frames.shape
            errors: list[Tensor] = []
            pair_details: list[dict] = []
            confidence_values: list[Tensor] = []
            feature_norm_values: list[Tensor] = []
            n_pairs = 0
            n_zero_overlap = 0
            for i in range(k_local):
                for j in range(i + 1, k_local):
                    n_pairs += 1
                    pair_imgs = torch.stack([frames[:, i], frames[:, j]], dim=1).cuda(non_blocking=True)
                    _, mask, score_map = super(MEt3R_MMD, self).forward(
                        pair_imgs,
                        return_overlap_mask=True,
                        return_score_map=True,
                        **kwargs,
                    )
                    pair_confidence = getattr(self, "_last_confidence_values", None)
                    if pair_confidence is not None and pair_confidence.numel() > 0:
                        confidence_values.append(
                            _subsample_values(
                                pair_confidence,
                                pair_details_max_samples,
                                seed=(i + 1) * 30_013 + (j + 1) * 3_011,
                            )
                        )
                    pair_feature_norms = getattr(self, "_last_feature_norm_values", None)
                    if pair_feature_norms is not None and pair_feature_norms.numel() > 0:
                        feature_norm_values.append(
                            _subsample_values(
                                pair_feature_norms,
                                pair_details_max_samples,
                                seed=(i + 1) * 40_009 + (j + 1) * 4_019,
                            )
                        )
                    pred_values = score_map[mask.bool()]
                    has_overlap = pred_values.numel() > 0
                    full_count = pred_values.numel()
                    full_mean = float(pred_values.mean()) if has_overlap else float("nan")
                    pred_values = pred_values.detach()
                    if sample_on_cpu:
                        pred_values = pred_values.to("cpu")

                    pair_detail_values = _subsample_values(
                        pred_values,
                        pair_details_max_samples,
                        seed=(i + 1) * 10_007 + (j + 1) * 1_009,
                    )
                    pair_details.append({
                        "i": i, "j": j,
                        "has_overlap": has_overlap,
                        "n_correspondences": full_count,
                        "mean_residual": full_mean,
                        "residuals_cpu": pair_detail_values if has_overlap else torch.empty(0),
                    })
                    if not has_overlap:
                        n_zero_overlap += 1
                        continue

                    pred_values = _subsample_values(
                        pred_values,
                        per_pair_max_samples,
                        seed=(i + 1) * 20_011 + (j + 1) * 2_027,
                    )

                    errors.append(pred_values)
            return errors, n_pairs, n_zero_overlap, pair_details, confidence_values, feature_norm_values

        err_list_pred, n_pairs, n_zero, pair_details, confidence_pred, feature_norm_pred = _collect_errors(images)
        self._last_overlap_stats = {"n_pairs": n_pairs, "n_zero_overlap": n_zero}
        self._last_pair_details = pair_details
        self._last_confidence_values = (
            torch.cat(confidence_pred, dim=0) if confidence_pred else torch.empty(0, dtype=torch.float32)
        )
        self._last_feature_norm_values = (
            torch.cat(feature_norm_pred, dim=0) if feature_norm_pred else torch.empty(0, dtype=torch.float32)
        )
        err_list_gt, _, _, _, _, _ = _collect_errors(gt_images) if gt_images is not None else ([], 0, 0, [], [], [])

        if not err_list_pred:
            nan = torch.tensor(float("nan"), device=device)
            if return_sigma:
                return (nan, nan)
            return (nan,)

        delta_pred = torch.cat(err_list_pred)  # (N_pred,)
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

        if gt_images is None:
            if return_sigma:
                return (mmd_pred.to(device), sigma_pred.to(device))
            return (mmd_pred.to(device),)

        # -----  include GT : MMD between two *empirical* distributions
        if not err_list_gt:
            if return_sigma:
                return (mmd_pred.to(device), sigma_pred.to(device))
            return (mmd_pred.to(device),)

        delta_gt = torch.cat(err_list_gt)  # (N_gt,)
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
                med = torch.median((delta_pred[:, None] - delta_gt[None, :]).abs())
            sigma = max(med.item(), 1e-6)
        kpq = torch.exp(-(delta_pred[:, None] - delta_gt[None, :]) ** 2 / (2 * sigma**2))

        mmd2 = mmd_pred  # first population term
        mmd2 += (torch.exp(-(delta_gt[:, None] - delta_gt[None, :])**2/(2*sigma**2)).sum() - delta_gt.numel()) / (delta_gt.numel() * (delta_gt.numel()-1))  # GT term
        mmd2 += -2 * kpq.mean() # cross-term

        if return_sigma:
            sigma_tensor = torch.tensor(sigma, device=delta_gt.device)
            return (mmd2.to(device), sigma_tensor.to(device))

        return (mmd2.to(device),)


class MEt3R_Energy(MEt3R):
    """
    MEt3R variant using Energy Distance instead of MMD.
    
    Energy distance is a kernel-free metric that doesn't require bandwidth tuning,
    making it more robust for comparing samples with different error magnitudes.
    
    Energy distance = 2*E[|X|] - E[|X-X'|]
    where X are errors and the reference is a point mass at 0 (perfect consistency).
    Higher values = worse consistency (errors far from 0).
    """
    def __init__(self,
                 *args,
                 per_pair_max_samples: Optional[int] = 128,
                 max_total_samples: Optional[int] = 4096,
                 sample_on_cpu: bool = True,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.per_pair_max_samples = per_pair_max_samples
        self.max_total_samples = max_total_samples
        self.sample_on_cpu = sample_on_cpu

    @torch.no_grad()
    def forward(self,
                images: Float[Tensor, "b k c h w"],
                **kwargs):
        """
        Args
        ----
        images     : predicted frames  (k may be > 2 - we will compare *all* pairs)
        
        Returns
        -------
        energy : Energy distance of the error distribution from delta_0.
        """
        b, k, c, h, w = images.shape
        device = images.device

        per_pair_max_samples = kwargs.pop("per_pair_max_samples", self.per_pair_max_samples)
        max_total_samples = kwargs.pop("max_total_samples", self.max_total_samples)
        sample_on_cpu = kwargs.pop("sample_on_cpu", self.sample_on_cpu)

        errors: list[Tensor] = []
        n_pairs = 0
        n_zero_overlap = 0
        for i in range(k):
            for j in range(i + 1, k):
                n_pairs += 1
                pair_imgs = torch.stack([images[:, i], images[:, j]], dim=1).cuda(non_blocking=True)
                _, mask, score_map = super(MEt3R_Energy, self).forward(
                    pair_imgs,
                    return_overlap_mask=True,
                    return_score_map=True,
                    **kwargs,
                )
                pred_values = score_map[mask.bool()]
                if pred_values.numel() == 0:
                    n_zero_overlap += 1
                    continue

                pred_values = pred_values.detach()
                if sample_on_cpu:
                    pred_values = pred_values.to("cpu")

                if per_pair_max_samples is not None and pred_values.numel() > per_pair_max_samples:
                    idx = torch.randperm(pred_values.numel(), device=pred_values.device)[:per_pair_max_samples]
                    pred_values = pred_values[idx]

                errors.append(pred_values)

        self._last_overlap_stats = {"n_pairs": n_pairs, "n_zero_overlap": n_zero_overlap}
        if not errors:
            return (torch.tensor(float("nan"), device=device),)

        delta = torch.cat(errors)
        if max_total_samples is not None and delta.numel() > max_total_samples:
            idx = torch.randperm(delta.numel(), device=delta.device)[:max_total_samples]
            delta = delta[idx]

        ed = energy_distance(delta)
        return (ed.to(device),)


class MEt3R_IMQ(MEt3R):
    """
    MEt3R using IMQ (Inverse Multiquadric) kernel MMD instead of RBF.
    
    The IMQ kernel K(x,y) = (c² + |x-y|²)^{-1/2} is bandwidth-free and more
    robust to outliers than RBF. This avoids the sigma selection problem.
    
    Higher values = worse consistency (errors far from 0).
    """
    def __init__(self,
                 *args,
                 c: float = 1.0,
                 per_pair_max_samples: Optional[int] = 128,
                 max_total_samples: Optional[int] = 4096,
                 sample_on_cpu: bool = True,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.c = c
        self.per_pair_max_samples = per_pair_max_samples
        self.max_total_samples = max_total_samples
        self.sample_on_cpu = sample_on_cpu

    @torch.no_grad()
    def forward(self,
                images: Float[Tensor, "b k c h w"],
                **kwargs):
        """
        Args
        ----
        images     : predicted frames  (k may be > 2 - we will compare *all* pairs)
        
        Returns
        -------
        mmd : IMQ kernel MMD of the error distribution from delta_0.
        """
        b, k, c, h, w = images.shape
        device = images.device

        per_pair_max_samples = kwargs.pop("per_pair_max_samples", self.per_pair_max_samples)
        max_total_samples = kwargs.pop("max_total_samples", self.max_total_samples)
        sample_on_cpu = kwargs.pop("sample_on_cpu", self.sample_on_cpu)

        errors: list[Tensor] = []
        n_pairs = 0
        n_zero_overlap = 0
        for i in range(k):
            for j in range(i + 1, k):
                n_pairs += 1
                pair_imgs = torch.stack([images[:, i], images[:, j]], dim=1).cuda(non_blocking=True)
                _, mask, score_map = super(MEt3R_IMQ, self).forward(
                    pair_imgs,
                    return_overlap_mask=True,
                    return_score_map=True,
                    **kwargs,
                )
                pred_values = score_map[mask.bool()]
                if pred_values.numel() == 0:
                    n_zero_overlap += 1
                    continue

                pred_values = pred_values.detach()
                if sample_on_cpu:
                    pred_values = pred_values.to("cpu")

                if per_pair_max_samples is not None and pred_values.numel() > per_pair_max_samples:
                    idx = torch.randperm(pred_values.numel(), device=pred_values.device)[:per_pair_max_samples]
                    pred_values = pred_values[idx]

                errors.append(pred_values)

        self._last_overlap_stats = {"n_pairs": n_pairs, "n_zero_overlap": n_zero_overlap}
        if not errors:
            return (torch.tensor(float("nan"), device=device),)

        delta = torch.cat(errors)
        if max_total_samples is not None and delta.numel() > max_total_samples:
            idx = torch.randperm(delta.numel(), device=delta.device)[:max_total_samples]
            delta = delta[idx]

        imq_mmd = mmd_imq(delta, c=self.c)
        return (imq_mmd.to(device),)
