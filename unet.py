"""
UNet Backbone for Diffusion Models
Implements a full UNet with time embeddings, self-attention, and residual blocks.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ─────────────────────────────────────────────
#  Time / Sinusoidal Embedding
# ─────────────────────────────────────────────

class SinusoidalPositionEmbeddings(nn.Module):
    """Classic sinusoidal timestep embeddings from DDPM."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


# ─────────────────────────────────────────────
#  Attention Block
# ─────────────────────────────────────────────

class AttentionBlock(nn.Module):
    """Multi-head self-attention for spatial features."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.scale = (channels // num_heads) ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (n d) h w -> (b n) (h w) d', n=self.num_heads)
        k = rearrange(k, 'b (n d) h w -> (b n) d (h w)', n=self.num_heads)
        v = rearrange(v, 'b (n d) h w -> (b n) (h w) d', n=self.num_heads)

        attn = torch.bmm(q, k) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.bmm(attn, v)

        out = rearrange(out, '(b n) (h w) d -> b (n d) h w', b=B, n=self.num_heads, h=H, w=W)
        return x + self.proj(out)


# ─────────────────────────────────────────────
#  Residual Block
# ─────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """ResNet-style block with time conditioning."""

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, dropout: float = 0.1):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
        )
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels * 2),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        self.res_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        # Time conditioning via scale-shift (AdaGN)
        time_out = self.time_mlp(time_emb)
        scale, shift = time_out.unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
        h = h * (1 + scale) + shift
        h = self.block2(h)
        return h + self.res_conv(x)


# ─────────────────────────────────────────────
#  Down / Up sampling blocks
# ─────────────────────────────────────────────

class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


# ─────────────────────────────────────────────
#  Full UNet
# ─────────────────────────────────────────────

class UNet(nn.Module):
    """
    Full UNet backbone for diffusion models.

    Args:
        in_channels:      Input image channels (e.g. 3 for RGB)
        model_channels:   Base feature channels
        out_channels:     Output channels (same as in_channels usually)
        channel_mult:     Multipliers per resolution level
        num_res_blocks:   ResBlocks per level
        attention_levels: Which levels get self-attention
        dropout:          Dropout probability
        num_classes:      If > 0, enables class conditioning
    """

    def __init__(
        self,
        in_channels: int = 3,
        model_channels: int = 128,
        out_channels: int = 3,
        channel_mult: tuple = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attention_levels: tuple = (2, 3),
        dropout: float = 0.1,
        num_classes: int = 0,
    ):
        super().__init__()
        self.num_classes = num_classes

        # Time embedding
        time_emb_dim = model_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(model_channels),
            nn.Linear(model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Class conditioning
        if num_classes > 0:
            self.class_emb = nn.Embedding(num_classes, time_emb_dim)

        # Input projection
        self.input_proj = nn.Conv2d(in_channels, model_channels, 3, padding=1)

        # ── Encoder (Downsampling) ──
        self.down_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        ch = model_channels
        input_block_chans = [ch]

        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks):
                block = ResidualBlock(ch, out_ch, time_emb_dim, dropout)
                if level in attention_levels:
                    self.down_blocks.append(nn.ModuleList([block, AttentionBlock(out_ch)]))
                else:
                    self.down_blocks.append(nn.ModuleList([block, nn.Identity()]))
                ch = out_ch
                input_block_chans.append(ch)

            if level < len(channel_mult) - 1:
                self.downsamplers.append(Downsample(ch))
                input_block_chans.append(ch)
            else:
                self.downsamplers.append(None)

        # ── Bottleneck ──
        self.mid_block1 = ResidualBlock(ch, ch, time_emb_dim, dropout)
        self.mid_attn = AttentionBlock(ch)
        self.mid_block2 = ResidualBlock(ch, ch, time_emb_dim, dropout)

        # ── Decoder (Upsampling) ──
        self.up_blocks = nn.ModuleList()
        self.upsamplers = nn.ModuleList()

        for level, mult in reversed(list(enumerate(channel_mult))):
            out_ch = model_channels * mult
            for i in range(num_res_blocks + 1):
                skip_ch = input_block_chans.pop()
                block = ResidualBlock(ch + skip_ch, out_ch, time_emb_dim, dropout)
                if level in attention_levels:
                    self.up_blocks.append(nn.ModuleList([block, AttentionBlock(out_ch)]))
                else:
                    self.up_blocks.append(nn.ModuleList([block, nn.Identity()]))
                ch = out_ch

            if level > 0:
                self.upsamplers.append(Upsample(ch))
            else:
                self.upsamplers.append(None)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        class_labels: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x:            Noisy input  [B, C, H, W]
            timesteps:    Diffusion timesteps [B]
            class_labels: Optional class IDs [B]
        Returns:
            Predicted noise [B, C, H, W]
        """
        # Embed time
        t_emb = self.time_mlp(timesteps)
        if self.num_classes > 0 and class_labels is not None:
            t_emb = t_emb + self.class_emb(class_labels)

        # Input
        h = self.input_proj(x)
        skips = [h]

        # Encoder
        down_idx = 0
        ds_idx = 0
        for i, (res, attn) in enumerate(self.down_blocks):
            h = res(h, t_emb)
            h = attn(h) if not isinstance(attn, nn.Identity) else h
            skips.append(h)

            # Downsample after each level's last block
            if (i + 1) % (len(self.down_blocks) // len(self.downsamplers)) == 0:
                if self.downsamplers[ds_idx] is not None:
                    h = self.downsamplers[ds_idx](h)
                    skips.append(h)
                ds_idx = min(ds_idx + 1, len(self.downsamplers) - 1)

        # Bottleneck
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        # Decoder
        up_idx = 0
        for i, (res, attn) in enumerate(self.up_blocks):
            h = torch.cat([h, skips.pop()], dim=1)
            h = res(h, t_emb)
            h = attn(h) if not isinstance(attn, nn.Identity) else h

            if (i + 1) % (len(self.up_blocks) // len(self.upsamplers)) == 0:
                if up_idx < len(self.upsamplers) and self.upsamplers[up_idx] is not None:
                    h = self.upsamplers[up_idx](h)
                up_idx = min(up_idx + 1, len(self.upsamplers) - 1)

        return self.output_proj(h)


# ─────────────────────────────────────────────
#  Model factory
# ─────────────────────────────────────────────

MODEL_CONFIGS = {
    "small": dict(model_channels=64,  channel_mult=(1, 2, 4),    num_res_blocks=2),
    "base":  dict(model_channels=128, channel_mult=(1, 2, 4, 8), num_res_blocks=2),
    "large": dict(model_channels=256, channel_mult=(1, 2, 4, 8), num_res_blocks=3),
}

def build_unet(size: str = "base", **kwargs) -> UNet:
    cfg = MODEL_CONFIGS[size].copy()
    cfg.update(kwargs)
    return UNet(**cfg)
