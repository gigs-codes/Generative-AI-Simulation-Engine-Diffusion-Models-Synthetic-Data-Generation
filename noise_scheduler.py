"""
Noise Schedulers for Diffusion Models.
Implements DDPM (cosine & linear betas) and DDIM deterministic sampling.
"""

import torch
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class DDPMOutput:
    noisy_sample: torch.Tensor
    noise: torch.Tensor
    pred_x0: Optional[torch.Tensor] = None


class DDPMScheduler:
    """
    Denoising Diffusion Probabilistic Models noise scheduler.
    Supports linear and cosine beta schedules.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_schedule: str = "cosine",   # "linear" | "cosine" | "sqrt_linear"
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        clip_sample: bool = True,
        clip_sample_range: float = 1.0,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range

        # Build beta schedule
        if beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        elif beta_schedule == "cosine":
            self.betas = self._cosine_beta_schedule(num_train_timesteps)
        elif beta_schedule == "sqrt_linear":
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps) ** 2
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        # Precompute diffusion parameters
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1), self.alphas_cumprod[:-1]])

        # For q(x_t | x_0)
        self.sqrt_alphas_cumprod = self.alphas_cumprod.sqrt()
        self.sqrt_one_minus_alphas_cumprod = (1.0 - self.alphas_cumprod).sqrt()

        # For posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            self.posterior_variance.clamp(min=1e-20)
        )
        self.posterior_mean_coef1 = (
            self.betas * self.alphas_cumprod_prev.sqrt() / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * self.alphas.sqrt() / (1.0 - self.alphas_cumprod)
        )

    @staticmethod
    def _cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
        """Improved cosine schedule from Nichol & Dhariwal 2021."""
        steps = T + 1
        x = torch.linspace(0, T, steps)
        alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * torch.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return betas.clamp(0, 0.9999)

    def _extract(self, a: torch.Tensor, t: torch.Tensor, shape: Tuple) -> torch.Tensor:
        """Extract values at timestep t and reshape to match image tensor."""
        batch = t.shape[0]
        out = a.to(t.device).gather(-1, t.long())
        return out.reshape(batch, *((1,) * (len(shape) - 1)))

    def add_noise(
        self,
        x_start: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
    ) -> DDPMOutput:
        """
        Forward diffusion: q(x_t | x_0).
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        if timesteps is None:
            timesteps = torch.randint(0, self.num_train_timesteps, (x_start.shape[0],))

        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, timesteps, x_start.shape)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape)

        noisy = sqrt_alpha * x_start + sqrt_one_minus * noise
        return DDPMOutput(noisy_sample=noisy, noise=noise)

    @torch.no_grad()
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        prediction_type: str = "epsilon",  # "epsilon" | "x_start" | "v"
    ) -> torch.Tensor:
        """
        Reverse diffusion step: p(x_{t-1} | x_t).
        """
        t = timestep
        prev_t = t - self.num_train_timesteps // 1000 if t > 0 else 0

        # Decode model output into x_0 prediction
        if prediction_type == "epsilon":
            alpha_prod_t = self.alphas_cumprod[t]
            x0_pred = (sample - (1 - alpha_prod_t) ** 0.5 * model_output) / alpha_prod_t ** 0.5
        elif prediction_type == "x_start":
            x0_pred = model_output
        elif prediction_type == "v":
            alpha_prod_t = self.alphas_cumprod[t]
            x0_pred = alpha_prod_t ** 0.5 * sample - (1 - alpha_prod_t) ** 0.5 * model_output
        else:
            raise ValueError(f"Unknown prediction type: {prediction_type}")

        if self.clip_sample:
            x0_pred = x0_pred.clamp(-self.clip_sample_range, self.clip_sample_range)

        # Compute posterior mean
        t_tensor = torch.tensor([t], device=sample.device)
        coef1 = self._extract(self.posterior_mean_coef1, t_tensor, sample.shape)
        coef2 = self._extract(self.posterior_mean_coef2, t_tensor, sample.shape)
        mean = coef1 * x0_pred + coef2 * sample

        if t == 0:
            return mean

        # Add posterior variance
        log_var = self._extract(self.posterior_log_variance_clipped, t_tensor, sample.shape)
        noise = torch.randn_like(sample)
        return mean + (0.5 * log_var).exp() * noise


class DDIMScheduler:
    """
    Denoising Diffusion Implicit Models scheduler (Song et al. 2020).
    Deterministic sampling with far fewer steps (e.g. 50 instead of 1000).
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_schedule: str = "cosine",
        eta: float = 0.0,          # 0 = fully deterministic, 1 = DDPM-like
        clip_sample: bool = True,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.eta = eta
        self.clip_sample = clip_sample

        # Reuse beta schedule from DDPM
        ddpm = DDPMScheduler(num_train_timesteps, beta_schedule)
        self.alphas_cumprod = ddpm.alphas_cumprod

        self.num_inference_steps: Optional[int] = None
        self.timesteps: Optional[torch.Tensor] = None

    def set_timesteps(self, num_inference_steps: int, device: str = "cpu"):
        """Set evenly-spaced inference timesteps."""
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // num_inference_steps
        self.timesteps = (
            torch.arange(0, num_inference_steps) * step_ratio
        ).flip(0).to(device)

    @torch.no_grad()
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        prev_timestep: Optional[int] = None,
    ) -> torch.Tensor:
        """Single DDIM reverse step."""
        t = timestep
        prev_t = prev_timestep if prev_timestep is not None else t - (self.num_train_timesteps // self.num_inference_steps)

        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_prev = self.alphas_cumprod[max(prev_t, 0)] if prev_t >= 0 else torch.tensor(1.0)

        # Predict x_0
        x0_pred = (sample - (1 - alpha_prod_t) ** 0.5 * model_output) / alpha_prod_t ** 0.5
        if self.clip_sample:
            x0_pred = x0_pred.clamp(-1, 1)

        # Direction towards x_t
        sigma = self.eta * ((1 - alpha_prod_prev) / (1 - alpha_prod_t) * (1 - alpha_prod_t / alpha_prod_prev)).sqrt()

        noise = torch.randn_like(sample) if self.eta > 0 else torch.zeros_like(sample)

        x_prev = (
            alpha_prod_prev ** 0.5 * x0_pred
            + (1 - alpha_prod_prev - sigma ** 2).clamp(min=0) ** 0.5 * model_output
            + sigma * noise
        )
        return x_prev
