"""
Distributed Training Engine
Handles multi-GPU training via PyTorch Accelerate with EMA, gradient scaling,
mixed precision, checkpointing, and WandB/TensorBoard logging.
"""

import os
import time
import math
import json
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.cuda.amp import GradScaler

try:
    from accelerate import Accelerator
    HAS_ACCELERATE = True
except ImportError:
    HAS_ACCELERATE = False

from tqdm import tqdm

from ..models.unet import UNet
from ..models.noise_scheduler import DDPMScheduler
from ..utils.ema import EMAModel
from ..utils.logger import Logger


class DiffusionTrainer:
    """
    Full distributed training engine for diffusion models.

    Features:
    - Mixed precision (fp16/bf16)
    - Multi-GPU via Accelerate or vanilla DDP
    - Exponential Moving Average (EMA) of weights
    - Gradient clipping & accumulation
    - Cosine LR schedule with warmup
    - Checkpointing every N steps
    - WandB + TensorBoard logging
    - Validation loss tracking
    """

    def __init__(self, config: dict):
        self.config = config
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(config.get("output_dir", "outputs"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ── Accelerator setup ──
        if HAS_ACCELERATE:
            self.accelerator = Accelerator(
                mixed_precision=config.get("mixed_precision", "fp16"),
                gradient_accumulation_steps=config.get("grad_accumulation", 1),
                log_with=config.get("log_with", "tensorboard"),
                project_dir=str(self.output_dir / "logs"),
            )
        else:
            self.accelerator = None

        # ── Model ──
        from ..models.unet import build_unet
        self.model = build_unet(
            size=config.get("model_size", "base"),
            in_channels=config.get("in_channels", 3),
            out_channels=config.get("in_channels", 3),
            num_classes=config.get("num_classes", 0),
        )

        # ── EMA ──
        self.ema = EMAModel(self.model, decay=config.get("ema_decay", 0.9999))

        # ── Noise Scheduler ──
        self.scheduler = DDPMScheduler(
            num_train_timesteps=config.get("num_timesteps", 1000),
            beta_schedule=config.get("beta_schedule", "cosine"),
        )

        # ── Optimizer ──
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.get("learning_rate", 2e-4),
            betas=(0.9, 0.999),
            weight_decay=config.get("weight_decay", 1e-4),
        )

        # ── Logger ──
        self.logger = Logger(
            log_dir=str(self.output_dir / "logs"),
            use_wandb=config.get("use_wandb", False),
            project_name=config.get("wandb_project", "genai-simulation-engine"),
            run_name=config.get("run_name", "ddpm-run"),
        )

        self.global_step = 0
        self.best_loss = float("inf")
        self.grad_scaler = GradScaler() if self.device == "cuda" else None

    def _build_scheduler(self, num_training_steps: int):
        warmup_steps = self.config.get("warmup_steps", 500)
        return OneCycleLR(
            self.optimizer,
            max_lr=self.config.get("learning_rate", 2e-4),
            total_steps=num_training_steps,
            pct_start=warmup_steps / num_training_steps,
        )

    def compute_loss(
        self,
        batch: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute diffusion training loss.
        L_simple = E[||eps - eps_theta(x_t, t)||^2]
        """
        noise = torch.randn_like(batch)
        timesteps = torch.randint(
            0, self.scheduler.num_train_timesteps, (batch.shape[0],), device=batch.device
        ).long()

        # Forward diffusion
        noisy = self.scheduler.add_noise(batch, noise, timesteps).noisy_sample

        # Predict noise
        predicted_noise = self.model(noisy, timesteps, class_labels)

        # MSE loss (simple objective from DDPM paper)
        loss = nn.functional.mse_loss(predicted_noise, noise)
        return loss

    def train(self, train_dataloader, val_dataloader=None, num_epochs: int = 100):
        """
        Main training loop.
        """
        num_training_steps = num_epochs * len(train_dataloader)
        lr_scheduler = self._build_scheduler(num_training_steps)

        # Wrap with accelerator if available
        if self.accelerator:
            self.model, self.optimizer, train_dataloader, lr_scheduler = \
                self.accelerator.prepare(self.model, self.optimizer, train_dataloader, lr_scheduler)

        self.model.to(self.device)

        print(f"\n{'='*60}")
        print(f"  Training DDPM")
        print(f"  Epochs:       {num_epochs}")
        print(f"  Steps:        {num_training_steps}")
        print(f"  Device:       {self.device}")
        print(f"  Batch size:   {self.config.get('batch_size', 32)}")
        print(f"  LR:           {self.config.get('learning_rate', 2e-4)}")
        print(f"  Model params: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"{'='*60}\n")

        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_start = time.time()

            pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch_idx, batch_data in enumerate(pbar):
                if isinstance(batch_data, (list, tuple)):
                    images, labels = batch_data[0], batch_data[1] if len(batch_data) > 1 else None
                else:
                    images, labels = batch_data, None

                images = images.to(self.device)
                if labels is not None:
                    labels = labels.to(self.device)

                # Forward + backward
                with torch.cuda.amp.autocast(enabled=self.grad_scaler is not None):
                    loss = self.compute_loss(images, labels)

                if self.grad_scaler:
                    self.grad_scaler.scale(loss).backward()
                    self.grad_scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                lr_scheduler.step()
                self.optimizer.zero_grad()
                self.ema.update(self.model)

                epoch_loss += loss.item()
                self.global_step += 1

                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}",
                })

                # Log every N steps
                if self.global_step % self.config.get("log_every", 100) == 0:
                    self.logger.log({
                        "train/loss": loss.item(),
                        "train/lr": lr_scheduler.get_last_lr()[0],
                        "train/step": self.global_step,
                    })

            avg_loss = epoch_loss / len(train_dataloader)
            epoch_time = time.time() - epoch_start

            # Validation
            val_loss = None
            if val_dataloader:
                val_loss = self._validate(val_dataloader)

            print(f"\nEpoch {epoch+1}/{num_epochs} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Val: {val_loss:.4f if val_loss else 'N/A'} | "
                  f"Time: {epoch_time:.1f}s")

            # Save checkpoint
            if (epoch + 1) % self.config.get("save_every", 10) == 0:
                self.save_checkpoint(epoch, avg_loss)

            if val_loss and val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_checkpoint(epoch, val_loss, is_best=True)

        print(f"\n✓ Training complete. Best loss: {self.best_loss:.4f}")
        self.logger.close()

    @torch.no_grad()
    def _validate(self, val_dataloader) -> float:
        self.model.eval()
        total_loss = 0.0
        for batch_data in val_dataloader:
            if isinstance(batch_data, (list, tuple)):
                images, labels = batch_data[0], batch_data[1] if len(batch_data) > 1 else None
            else:
                images, labels = batch_data, None
            images = images.to(self.device)
            if labels is not None:
                labels = labels.to(self.device)
            loss = self.compute_loss(images, labels)
            total_loss += loss.item()
        return total_loss / len(val_dataloader)

    def save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        """Save model + EMA + optimizer state."""
        name = "best_model.pt" if is_best else f"checkpoint_epoch_{epoch+1:04d}.pt"
        path = self.output_dir / "checkpoints" / name
        path.parent.mkdir(parents=True, exist_ok=True)

        unwrapped = self.accelerator.unwrap_model(self.model) if self.accelerator else self.model
        torch.save({
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": unwrapped.state_dict(),
            "ema_state_dict": self.ema.shadow_params,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": loss,
            "config": self.config,
        }, path)
        print(f"  → Saved checkpoint: {path}")

    def load_checkpoint(self, path: str):
        """Resume training from checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.global_step = ckpt.get("global_step", 0)
        self.best_loss = ckpt.get("loss", float("inf"))
        if "ema_state_dict" in ckpt:
            self.ema.shadow_params = ckpt["ema_state_dict"]
        print(f"✓ Resumed from {path} (step {self.global_step})")
