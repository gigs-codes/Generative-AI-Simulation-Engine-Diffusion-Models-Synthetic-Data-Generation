"""
Variational Autoencoder (VAE) for Latent Diffusion Models.
Encodes images to a compact latent space; diffusion operates in latent space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple


@dataclass
class VAEOutput:
    z: torch.Tensor          # Sampled latent
    mu: torch.Tensor         # Mean
    log_var: torch.Tensor    # Log variance
    recon: torch.Tensor      # Reconstruction
    kl_loss: torch.Tensor    # KL divergence
    recon_loss: torch.Tensor # Reconstruction loss


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)


class Encoder(nn.Module):
    """Encodes images to (mu, log_var) in latent space."""

    def __init__(self, in_channels: int, latent_channels: int, base_channels: int = 128):
        super().__init__()
        ch = base_channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, ch, 3, padding=1),
            ResBlock(ch),
            nn.Conv2d(ch, ch * 2, 4, stride=2, padding=1),   # /2
            ResBlock(ch * 2),
            nn.Conv2d(ch * 2, ch * 4, 4, stride=2, padding=1), # /4
            ResBlock(ch * 4),
            nn.Conv2d(ch * 4, ch * 4, 4, stride=2, padding=1), # /8
            ResBlock(ch * 4),
            nn.GroupNorm(8, ch * 4),
            nn.SiLU(),
        )
        self.to_mu = nn.Conv2d(ch * 4, latent_channels, 1)
        self.to_log_var = nn.Conv2d(ch * 4, latent_channels, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        return self.to_mu(h), self.to_log_var(h)


class Decoder(nn.Module):
    """Decodes latent vectors back to image space."""

    def __init__(self, out_channels: int, latent_channels: int, base_channels: int = 128):
        super().__init__()
        ch = base_channels
        self.net = nn.Sequential(
            nn.Conv2d(latent_channels, ch * 4, 3, padding=1),
            ResBlock(ch * 4),
            nn.ConvTranspose2d(ch * 4, ch * 4, 4, stride=2, padding=1),  # ×2
            ResBlock(ch * 4),
            nn.ConvTranspose2d(ch * 4, ch * 2, 4, stride=2, padding=1),  # ×4
            ResBlock(ch * 2),
            nn.ConvTranspose2d(ch * 2, ch, 4, stride=2, padding=1),      # ×8
            ResBlock(ch),
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class VAE(nn.Module):
    """
    Variational Autoencoder.

    Args:
        in_channels:      Image channels (3 for RGB)
        latent_channels:  Latent space depth (e.g. 4)
        base_channels:    Feature width
        kl_weight:        Beta for KL annealing (beta-VAE)
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 4,
        base_channels: int = 128,
        kl_weight: float = 1e-4,
    ):
        super().__init__()
        self.kl_weight = kl_weight
        self.latent_channels = latent_channels
        self.encoder = Encoder(in_channels, latent_channels, base_channels)
        self.decoder = Decoder(in_channels, latent_channels, base_channels)

        # Learnable scaling factor (used in LDMs to scale latents to ~unit variance)
        self.scale_factor = nn.Parameter(torch.ones(1))

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterisation trick: z = mu + eps * std."""
        if self.training:
            std = (0.5 * log_var).exp()
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu  # Deterministic at inference

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, log_var = self.encoder(x)
        return mu, log_var

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z / self.scale_factor)

    def forward(self, x: torch.Tensor) -> VAEOutput:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z)

        # KL divergence: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        recon_loss = F.mse_loss(recon, x)

        return VAEOutput(
            z=z * self.scale_factor,
            mu=mu,
            log_var=log_var,
            recon=recon,
            kl_loss=kl_loss,
            recon_loss=recon_loss,
        )

    def total_loss(self, output: VAEOutput) -> torch.Tensor:
        return output.recon_loss + self.kl_weight * output.kl_loss

    @torch.no_grad()
    def encode_to_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image to scaled latent (for use in LDM training)."""
        mu, _ = self.encode(x)
        return mu * self.scale_factor

    @torch.no_grad()
    def decode_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from scaled latent back to image space."""
        return self.decode(z)
