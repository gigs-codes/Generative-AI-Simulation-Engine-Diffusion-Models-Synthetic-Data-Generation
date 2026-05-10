"""
Synthetic Data Generation Pipeline
Generates and augments datasets using trained diffusion models.
Supports image, tabular, and multimodal synthetic data.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from tqdm import tqdm
from PIL import Image
import json
from dataclasses import dataclass, asdict


@dataclass
class GenerationConfig:
    num_samples: int = 1000
    batch_size: int = 32
    image_size: int = 32
    guidance_scale: float = 3.0
    num_inference_steps: int = 50
    seed: Optional[int] = 42
    output_dir: str = "outputs/samples"
    save_metadata: bool = True
    augmentation_factor: int = 1  # Generate N× the original dataset size


class SyntheticDataGenerator:
    """
    High-level generator for synthetic datasets.
    Wraps diffusion pipelines to produce large-scale synthetic datasets.
    """

    def __init__(self, pipeline, config: GenerationConfig):
        self.pipeline = pipeline
        self.config = config
        self.device = pipeline.device

    def generate_dataset(
        self,
        class_labels: Optional[List[int]] = None,
        num_classes: Optional[int] = None,
        balanced: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate a complete synthetic image dataset.

        Args:
            class_labels:  Specific classes to generate. None = unconditional.
            num_classes:   Total number of classes for balanced generation.
            balanced:      Generate equal samples per class.

        Returns:
            Dict with 'images' and optionally 'labels'.
        """
        cfg = self.config
        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        all_images = []
        all_labels = []
        metadata = []

        if class_labels is None and num_classes is None:
            # Unconditional generation
            all_images, _ = self._generate_batch_loop(cfg.num_samples)
        else:
            # Conditional generation
            classes = class_labels if class_labels else list(range(num_classes))
            per_class = cfg.num_samples // len(classes) if balanced else cfg.num_samples

            for cls_id in tqdm(classes, desc="Generating classes"):
                labels_tensor = torch.full((per_class,), cls_id, dtype=torch.long)
                images, batch_meta = self._generate_batch_loop(per_class, labels_tensor)
                all_images.extend(images)
                all_labels.extend([cls_id] * per_class)
                metadata.extend(batch_meta)

        # Save to disk
        images_tensor = torch.stack(all_images) if isinstance(all_images[0], torch.Tensor) else torch.tensor(all_images)
        self._save_images(images_tensor, output_dir)

        if cfg.save_metadata:
            meta_path = output_dir / "metadata.json"
            with open(meta_path, "w") as f:
                json.dump({
                    "num_samples": len(all_images),
                    "image_size": cfg.image_size,
                    "guidance_scale": cfg.guidance_scale,
                    "inference_steps": cfg.num_inference_steps,
                    "seed": cfg.seed,
                    "classes": list(set(all_labels)) if all_labels else None,
                }, f, indent=2)

        result = {"images": images_tensor}
        if all_labels:
            result["labels"] = torch.tensor(all_labels, dtype=torch.long)

        print(f"✓ Generated {len(all_images)} synthetic samples → {output_dir}")
        return result

    def _generate_batch_loop(
        self,
        total: int,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[List, List]:
        """Loop over batches to generate `total` samples."""
        cfg = self.config
        all_images = []
        metadata = []
        generated = 0

        while generated < total:
            batch_n = min(cfg.batch_size, total - generated)
            batch_labels = labels[generated:generated + batch_n] if labels is not None else None
            if batch_labels is not None:
                batch_labels = batch_labels.to(self.device)

            images = self.pipeline.generate(
                batch_size=batch_n,
                class_labels=batch_labels,
                num_inference_steps=cfg.num_inference_steps,
                guidance_scale=cfg.guidance_scale,
                seed=cfg.seed + generated if cfg.seed else None,
                show_progress=False,
            )
            all_images.append(images.cpu())
            metadata.append({"batch_start": generated, "batch_size": batch_n})
            generated += batch_n

        return torch.cat(all_images, dim=0).unbind(0), metadata

    def _save_images(self, images: torch.Tensor, output_dir: Path):
        """Save tensor of images [N, C, H, W] as PNGs."""
        imgs_np = (images.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
        for i, img in enumerate(imgs_np):
            Image.fromarray(img).save(output_dir / f"sample_{i:06d}.png")

    def augment_dataset(
        self,
        original_images: torch.Tensor,
        original_labels: Optional[torch.Tensor] = None,
        augmentation_factor: int = 2,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Augment an existing dataset with synthetic samples.
        Generates (augmentation_factor - 1) × original_size new samples.

        Returns:
            Augmented images and labels (original + synthetic).
        """
        n_orig = len(original_images)
        n_synthetic = n_orig * (augmentation_factor - 1)

        print(f"Augmenting: {n_orig} original → {n_orig + n_synthetic} total")

        labels = None
        if original_labels is not None:
            # Sample labels proportionally from original distribution
            label_dist = torch.bincount(original_labels)
            probs = label_dist.float() / label_dist.sum()
            sampled_labels = torch.multinomial(probs, n_synthetic, replacement=True)
            labels = sampled_labels.to(self.device)

        synthetic_images, _ = self._generate_batch_loop(n_synthetic, labels)
        synthetic_tensor = torch.stack(synthetic_images)

        aug_images = torch.cat([original_images.cpu(), synthetic_tensor.cpu()], dim=0)
        aug_labels = None
        if original_labels is not None and labels is not None:
            aug_labels = torch.cat([original_labels.cpu(), labels.cpu()], dim=0)

        return aug_images, aug_labels


class TabularSyntheticGenerator:
    """
    Generates synthetic tabular data using a score-based diffusion model.
    Useful for privacy-preserving data augmentation of CSV/DataFrame datasets.
    """

    def __init__(self, model, device: str = "cuda"):
        self.model = model.to(device)
        self.device = device
        self.scaler = None
        self.feature_names: List[str] = []

    def fit(self, df: pd.DataFrame, target_col: Optional[str] = None):
        """Fit normalizer on real data."""
        from sklearn.preprocessing import StandardScaler
        self.feature_names = df.columns.tolist()
        self.target_col = target_col
        self.scaler = StandardScaler()
        data = df.values.astype(np.float32)
        self.scaler.fit(data)
        print(f"✓ Fitted on {len(df)} rows × {len(self.feature_names)} features")

    @torch.no_grad()
    def generate(self, n_samples: int = 1000, num_steps: int = 100) -> pd.DataFrame:
        """
        Generate n_samples rows of synthetic tabular data.

        Returns:
            DataFrame with same column names as training data.
        """
        assert self.scaler is not None, "Call fit() before generate()"
        n_features = len(self.feature_names)

        # Start from Gaussian noise
        x = torch.randn(n_samples, n_features, device=self.device)

        # Langevin dynamics / reverse SDE
        step_size = 0.01
        for _ in tqdm(range(num_steps), desc="Tabular generation"):
            score = self.model(x)
            noise = torch.randn_like(x) * np.sqrt(2 * step_size)
            x = x + step_size * score + noise

        # Inverse-normalize
        x_np = x.cpu().numpy()
        x_inv = self.scaler.inverse_transform(x_np)
        return pd.DataFrame(x_inv, columns=self.feature_names)

    def compute_fidelity(
        self, real_df: pd.DataFrame, synthetic_df: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Compute statistical fidelity metrics between real and synthetic data.
        """
        metrics = {}
        for col in real_df.columns:
            if col not in synthetic_df.columns:
                continue
            r = real_df[col].dropna().values
            s = synthetic_df[col].dropna().values
            metrics[f"{col}_mean_diff"] = abs(r.mean() - s.mean())
            metrics[f"{col}_std_diff"] = abs(r.std() - s.std())
            # Kolmogorov-Smirnov test
            from scipy.stats import ks_2samp
            ks_stat, _ = ks_2samp(r, s)
            metrics[f"{col}_ks_stat"] = ks_stat
        return metrics
