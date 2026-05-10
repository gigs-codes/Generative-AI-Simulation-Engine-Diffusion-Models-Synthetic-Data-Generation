"""
DDPM Diffusion Pipeline
End-to-end sampling pipeline for Denoising Diffusion Probabilistic Models.
"""

import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Optional, List, Union
from PIL import Image

from ..models.unet import UNet, build_unet
from ..models.noise_scheduler import DDPMScheduler, DDIMScheduler


class DDPMPipeline:
    """
    Full DDPM generation pipeline.

    Usage:
        pipeline = DDPMPipeline.from_config(config)
        images = pipeline.generate(batch_size=16)
        pipeline.save_images(images, "outputs/samples/")
    """

    def __init__(
        self,
        unet: UNet,
        scheduler: DDPMScheduler,
        image_size: int = 32,
        device: str = "cuda",
    ):
        self.unet = unet.to(device)
        self.scheduler = scheduler
        self.image_size = image_size
        self.device = device

    @classmethod
    def from_config(cls, config: dict, device: str = "cuda") -> "DDPMPipeline":
        unet = build_unet(
            size=config.get("model_size", "base"),
            in_channels=config.get("in_channels", 3),
            out_channels=config.get("in_channels", 3),
            num_classes=config.get("num_classes", 0),
        )
        scheduler = DDPMScheduler(
            num_train_timesteps=config.get("num_timesteps", 1000),
            beta_schedule=config.get("beta_schedule", "cosine"),
        )
        return cls(unet, scheduler, config.get("image_size", 32), device)

    @torch.no_grad()
    def generate(
        self,
        batch_size: int = 1,
        class_labels: Optional[torch.Tensor] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: float = 1.0,
        seed: Optional[int] = None,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """
        Sample images using the reverse diffusion process.

        Args:
            batch_size:           Number of images to generate
            class_labels:         Optional class conditioning labels
            num_inference_steps:  Override number of denoising steps
            guidance_scale:       Classifier-free guidance scale (>1 for conditioning)
            seed:                 Random seed for reproducibility
            show_progress:        Show tqdm progress bar

        Returns:
            Tensor of shape [B, C, H, W] in range [0, 1]
        """
        if seed is not None:
            torch.manual_seed(seed)

        self.unet.eval()
        steps = num_inference_steps or self.scheduler.num_train_timesteps
        C = self.unet.input_proj.in_channels if hasattr(self.unet, 'input_proj') else 3

        # Start from pure noise
        x = torch.randn(batch_size, C, self.image_size, self.image_size, device=self.device)

        timesteps = list(reversed(range(0, self.scheduler.num_train_timesteps)))
        if num_inference_steps and num_inference_steps < self.scheduler.num_train_timesteps:
            step_ratio = self.scheduler.num_train_timesteps // num_inference_steps
            timesteps = list(reversed(range(0, self.scheduler.num_train_timesteps, step_ratio)))

        pbar = tqdm(timesteps, desc="Sampling", disable=not show_progress)
        for t in pbar:
            t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)

            # Classifier-free guidance: run conditional + unconditional
            if guidance_scale > 1.0 and class_labels is not None:
                noise_pred_cond = self.unet(x, t_batch, class_labels)
                noise_pred_uncond = self.unet(x, t_batch)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = self.unet(x, t_batch, class_labels)

            x = self.scheduler.step(noise_pred, t, x)

        # Normalize from [-1, 1] to [0, 1]
        x = (x.clamp(-1, 1) + 1) / 2
        return x

    def save_images(self, images: torch.Tensor, output_dir: str, prefix: str = "sample"):
        """Save generated images to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        images_np = (images.cpu().permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
        saved = []
        for i, img_np in enumerate(images_np):
            path = output_dir / f"{prefix}_{i:04d}.png"
            Image.fromarray(img_np).save(path)
            saved.append(str(path))
        return saved

    def make_grid(self, images: torch.Tensor, nrow: int = 8) -> Image.Image:
        """Arrange images into a grid."""
        from torchvision.utils import make_grid
        grid = make_grid(images, nrow=nrow, normalize=False, padding=2)
        grid_np = (grid.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(grid_np)

    def load_checkpoint(self, path: str):
        """Load model weights from checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        if "model_state_dict" in ckpt:
            self.unet.load_state_dict(ckpt["model_state_dict"])
        else:
            self.unet.load_state_dict(ckpt)
        print(f"✓ Loaded checkpoint from {path}")


class DDIMPipeline(DDPMPipeline):
    """
    DDIM pipeline — same as DDPM but uses the deterministic DDIM sampler.
    Much faster inference: 50 steps instead of 1000.
    """

    def __init__(self, unet: UNet, ddim_scheduler: DDIMScheduler, image_size: int, device: str):
        self.unet = unet.to(device)
        self.scheduler = ddim_scheduler
        self.image_size = image_size
        self.device = device

    @torch.no_grad()
    def generate(
        self,
        batch_size: int = 1,
        class_labels: Optional[torch.Tensor] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
        seed: Optional[int] = None,
        show_progress: bool = True,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """Fast DDIM sampling with configurable number of steps."""
        if seed is not None:
            torch.manual_seed(seed)

        self.unet.eval()
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        C = 3  # RGB

        x = torch.randn(batch_size, C, self.image_size, self.image_size, device=self.device)

        timesteps = self.scheduler.timesteps
        pbar = tqdm(range(len(timesteps)), desc=f"DDIM ({num_inference_steps} steps)", disable=not show_progress)

        for i in pbar:
            t = int(timesteps[i])
            prev_t = int(timesteps[i + 1]) if i + 1 < len(timesteps) else -1
            t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)

            noise_pred = self.unet(x, t_batch, class_labels)
            x = self.scheduler.step(noise_pred, t, x, prev_t)

        return (x.clamp(-1, 1) + 1) / 2
