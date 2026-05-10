"""
Exponential Moving Average (EMA) for Model Weights.
Keeps a smoothed copy of model weights for better generation quality.
"""

import copy
import torch
import torch.nn as nn
from typing import Union


class EMAModel:
    """
    Maintains an exponential moving average of model parameters.
    EMA weights typically produce higher-quality samples.

    Usage:
        ema = EMAModel(model, decay=0.9999)
        for batch in dataloader:
            loss = model(batch)
            loss.backward()
            optimizer.step()
            ema.update(model)          # after every optimizer step

        # At inference time:
        ema.apply_shadow(model)        # swap in EMA weights
        samples = model.generate(...)
        ema.restore(model)             # restore original weights
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999, warmup_steps: int = 100):
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.step = 0

        # Maintain shadow (EMA) copy of parameters
        self.shadow_params = {
            name: param.clone().detach().float()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        # Store backup of original params (for restore)
        self.backup_params = {}

    def get_decay(self) -> float:
        """Warmup: use smaller decay at the start of training."""
        value = min(self.decay, (1 + self.step) / (10 + self.step))
        return value

    @torch.no_grad()
    def update(self, model: nn.Module):
        """Update EMA weights after each optimizer step."""
        self.step += 1
        decay = self.get_decay()

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            assert name in self.shadow_params, f"Unknown param: {name}"
            self.shadow_params[name].mul_(decay).add_(
                param.data.float(), alpha=1.0 - decay
            )

    def apply_shadow(self, model: nn.Module):
        """Swap model weights with EMA shadow weights (saves originals for restore)."""
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            assert name in self.shadow_params
            self.backup_params[name] = param.data.clone()
            param.data.copy_(self.shadow_params[name].to(param.device).to(param.dtype))

    def restore(self, model: nn.Module):
        """Restore original model weights after inference with EMA."""
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            assert name in self.backup_params, "Call apply_shadow() before restore()"
            param.data.copy_(self.backup_params[name])
        self.backup_params = {}

    def state_dict(self) -> dict:
        return {"shadow_params": self.shadow_params, "step": self.step}

    def load_state_dict(self, state: dict):
        self.shadow_params = state["shadow_params"]
        self.step = state.get("step", 0)
