"""
Training Logger
Unified logging interface for TensorBoard, WandB, and console output.
"""

import os
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional


class Logger:
    """
    Unified logger supporting TensorBoard and Weights & Biases.
    Falls back gracefully if libraries aren't installed.
    """

    def __init__(
        self,
        log_dir: str = "outputs/logs",
        use_wandb: bool = False,
        project_name: str = "genai-simulation-engine",
        run_name: Optional[str] = None,
        config: Optional[dict] = None,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb
        self._tb_writer = None
        self._wandb_run = None
        self.start_time = time.time()

        # TensorBoard
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._tb_writer = SummaryWriter(log_dir=str(self.log_dir))
            print(f"  TensorBoard logging → {self.log_dir}")
        except ImportError:
            pass

        # WandB
        if use_wandb:
            try:
                import wandb
                self._wandb_run = wandb.init(
                    project=project_name,
                    name=run_name or f"run-{int(time.time())}",
                    config=config or {},
                )
                print(f"  WandB logging → {project_name}/{run_name}")
            except Exception as e:
                print(f"  WandB init failed: {e}")
                self.use_wandb = False

        # JSON fallback log
        self._json_log = self.log_dir / "metrics.jsonl"

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log a dict of metrics at the given step."""
        # TensorBoard
        if self._tb_writer:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self._tb_writer.add_scalar(key, value, global_step=step)

        # WandB
        if self.use_wandb and self._wandb_run:
            try:
                import wandb
                wandb.log(metrics, step=step)
            except Exception:
                pass

        # JSON fallback
        with open(self._json_log, "a") as f:
            record = {"step": step, "time": time.time() - self.start_time, **metrics}
            f.write(json.dumps(record) + "\n")

    def log_image(self, tag: str, image_tensor, step: Optional[int] = None):
        """Log image to TensorBoard/WandB."""
        if self._tb_writer:
            self._tb_writer.add_image(tag, image_tensor, global_step=step)
        if self.use_wandb and self._wandb_run:
            try:
                import wandb
                import numpy as np
                img_np = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                wandb.log({tag: wandb.Image(img_np)}, step=step)
            except Exception:
                pass

    def log_histogram(self, tag: str, values, step: Optional[int] = None):
        """Log histogram of values."""
        if self._tb_writer:
            self._tb_writer.add_histogram(tag, values, global_step=step)

    def close(self):
        if self._tb_writer:
            self._tb_writer.close()
        if self.use_wandb and self._wandb_run:
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass
