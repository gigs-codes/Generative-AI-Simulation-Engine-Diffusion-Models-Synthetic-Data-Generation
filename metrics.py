"""
Evaluation Metrics for Generative Models
Fréchet Inception Distance (FID), Inception Score (IS), and LPIPS.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple
from tqdm import tqdm


class InceptionFeatureExtractor(nn.Module):
    """
    Extracts features from Inception-v3 for FID computation.
    Uses the pool3 layer (2048-dim features).
    """

    def __init__(self):
        super().__init__()
        try:
            from torchvision.models import inception_v3, Inception_V3_Weights
            self.inception = inception_v3(weights=Inception_V3_Weights.DEFAULT)
            self.inception.fc = nn.Identity()
            self.inception.eval()
        except Exception:
            self.inception = None
            print("Warning: Inception-v3 not available. FID computation disabled.")

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.inception is None:
            return torch.zeros(x.shape[0], 2048)
        # Resize to 299×299 for Inception
        import torch.nn.functional as F
        if x.shape[-1] != 299:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        # Normalize to Inception range
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        return self.inception(x)


def compute_fid(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    device: str = "cuda",
    batch_size: int = 64,
) -> float:
    """
    Compute Fréchet Inception Distance between real and generated images.
    Lower is better; state-of-the-art models achieve FID < 5.

    Args:
        real_images: Real images [N, C, H, W] in [0, 1]
        fake_images: Generated images [N, C, H, W] in [0, 1]
        device:      Compute device
        batch_size:  Batch size for feature extraction

    Returns:
        FID score (float)
    """
    extractor = InceptionFeatureExtractor().to(device)

    def extract_features(images: torch.Tensor) -> np.ndarray:
        feats = []
        for i in tqdm(range(0, len(images), batch_size), desc="Extracting features"):
            batch = images[i:i+batch_size].to(device)
            feat = extractor(batch).cpu().numpy()
            feats.append(feat)
        return np.concatenate(feats, axis=0)

    real_feats = extract_features(real_images)
    fake_feats = extract_features(fake_images)

    # Compute statistics
    mu1, sigma1 = real_feats.mean(0), np.cov(real_feats, rowvar=False)
    mu2, sigma2 = fake_feats.mean(0), np.cov(fake_feats, rowvar=False)

    # FID = ||mu1 - mu2||^2 + Tr(Sigma1 + Sigma2 - 2*sqrt(Sigma1@Sigma2))
    from scipy.linalg import sqrtm
    diff = mu1 - mu2
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return float(fid)


def compute_inception_score(
    images: torch.Tensor,
    device: str = "cuda",
    splits: int = 10,
    batch_size: int = 64,
) -> Tuple[float, float]:
    """
    Compute Inception Score (IS).
    Higher is better; good models achieve IS > 8 on CIFAR-10.

    Returns:
        (mean IS, std IS) across splits
    """
    try:
        from torchvision.models import inception_v3, Inception_V3_Weights
        model = inception_v3(weights=Inception_V3_Weights.DEFAULT).to(device)
        model.eval()
    except Exception:
        print("Warning: Inception-v3 not available.")
        return 0.0, 0.0

    preds = []
    with torch.no_grad():
        for i in tqdm(range(0, len(images), batch_size), desc="Computing IS"):
            batch = images[i:i+batch_size].to(device)
            import torch.nn.functional as F
            if batch.shape[-1] != 299:
                batch = F.interpolate(batch, (299, 299), mode='bilinear', align_corners=False)
            out = torch.nn.functional.softmax(model(batch), dim=1)
            preds.append(out.cpu().numpy())

    preds = np.concatenate(preds, axis=0)

    # Split into groups and compute KL divergence
    scores = []
    chunk_size = len(preds) // splits
    for k in range(splits):
        p_yx = preds[k * chunk_size:(k + 1) * chunk_size]
        p_y = p_yx.mean(0, keepdims=True)
        kl = p_yx * (np.log(p_yx + 1e-16) - np.log(p_y + 1e-16))
        kl = kl.sum(1).mean()
        scores.append(np.exp(kl))

    return float(np.mean(scores)), float(np.std(scores))


def compute_lpips(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    device: str = "cuda",
) -> float:
    """
    Compute Learned Perceptual Image Patch Similarity (LPIPS).
    Lower is better (more perceptually similar).
    """
    try:
        import lpips
        loss_fn = lpips.LPIPS(net='alex').to(device)
        # Rescale from [0, 1] to [-1, 1]
        real = (real_images.to(device) * 2 - 1).clamp(-1, 1)
        fake = (fake_images.to(device) * 2 - 1).clamp(-1, 1)
        with torch.no_grad():
            dist = loss_fn(real, fake)
        return float(dist.mean().cpu())
    except ImportError:
        print("Warning: lpips not installed. Run: pip install lpips")
        return -1.0


class EvaluationSuite:
    """
    Runs all evaluation metrics and returns a summary dict.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device

    def evaluate(
        self,
        real_images: torch.Tensor,
        fake_images: torch.Tensor,
        compute_fid: bool = True,
        compute_is: bool = True,
        compute_lpips_score: bool = True,
    ) -> dict:
        results = {}

        if compute_fid:
            print("Computing FID...")
            results["fid"] = compute_fid(real_images, fake_images, self.device)

        if compute_is:
            print("Computing IS...")
            is_mean, is_std = compute_inception_score(fake_images, self.device)
            results["is_mean"] = is_mean
            results["is_std"] = is_std

        if compute_lpips_score:
            print("Computing LPIPS...")
            results["lpips"] = compute_lpips(real_images, fake_images, self.device)

        print("\n📊 Evaluation Results:")
        for k, v in results.items():
            print(f"   {k:15s}: {v:.4f}")

        return results
