"""
Unit Tests for GenAI Simulation Engine
Run: pytest tests/ -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
import numpy as np

from src.models.unet import UNet, build_unet
from src.models.vae import VAE
from src.models.noise_scheduler import DDPMScheduler, DDIMScheduler
from src.utils.ema import EMAModel


# ─────────────────────────────────────────────
#  UNet Tests
# ─────────────────────────────────────────────

class TestUNet:
    def test_forward_small(self):
        model = build_unet("small", in_channels=3, out_channels=3)
        x = torch.randn(2, 3, 32, 32)
        t = torch.randint(0, 1000, (2,))
        out = model(x, t)
        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"

    def test_forward_base(self):
        model = build_unet("base", in_channels=3, out_channels=3)
        x = torch.randn(4, 3, 32, 32)
        t = torch.randint(0, 1000, (4,))
        out = model(x, t)
        assert out.shape == x.shape

    def test_class_conditioning(self):
        model = build_unet("small", in_channels=3, out_channels=3, num_classes=10)
        x = torch.randn(2, 3, 32, 32)
        t = torch.randint(0, 1000, (2,))
        labels = torch.randint(0, 10, (2,))
        out = model(x, t, labels)
        assert out.shape == x.shape

    def test_different_sizes(self):
        for size in [16, 32, 64]:
            model = build_unet("small")
            x = torch.randn(1, 3, size, size)
            t = torch.randint(0, 1000, (1,))
            out = model(x, t)
            assert out.shape == x.shape, f"Failed at size {size}"

    def test_gradient_flow(self):
        model = build_unet("small")
        x = torch.randn(2, 3, 32, 32)
        t = torch.randint(0, 1000, (2,))
        out = model(x, t)
        loss = out.mean()
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


# ─────────────────────────────────────────────
#  VAE Tests
# ─────────────────────────────────────────────

class TestVAE:
    def test_forward(self):
        vae = VAE(in_channels=3, latent_channels=4, base_channels=32)
        x = torch.randn(2, 3, 64, 64)
        out = vae(x)
        assert out.z.shape[1] == 4
        assert out.recon.shape == x.shape

    def test_encode_decode(self):
        vae = VAE(in_channels=3, latent_channels=4, base_channels=32)
        x = torch.randn(2, 3, 64, 64)
        z = vae.encode_to_latent(x)
        recon = vae.decode_from_latent(z)
        assert recon.shape == x.shape

    def test_kl_loss_nonnegative(self):
        vae = VAE(in_channels=3, latent_channels=4, base_channels=32)
        x = torch.randn(2, 3, 64, 64)
        out = vae(x)
        assert out.kl_loss.item() >= 0

    def test_recon_loss_nonnegative(self):
        vae = VAE(in_channels=3, latent_channels=4, base_channels=32)
        x = torch.randn(2, 3, 64, 64)
        out = vae(x)
        assert out.recon_loss.item() >= 0


# ─────────────────────────────────────────────
#  Noise Scheduler Tests
# ─────────────────────────────────────────────

class TestDDPMScheduler:
    def test_betas_in_range(self):
        sched = DDPMScheduler(num_train_timesteps=1000, beta_schedule="cosine")
        assert sched.betas.min() >= 0
        assert sched.betas.max() <= 1

    def test_add_noise_shape(self):
        sched = DDPMScheduler()
        x = torch.randn(4, 3, 32, 32)
        t = torch.randint(0, 1000, (4,))
        out = sched.add_noise(x, timesteps=t)
        assert out.noisy_sample.shape == x.shape
        assert out.noise.shape == x.shape

    def test_noisy_sample_is_noisy(self):
        sched = DDPMScheduler()
        x = torch.zeros(2, 3, 32, 32)
        t = torch.full((2,), 999)  # max noise
        out = sched.add_noise(x, timesteps=t)
        # At t=999 almost all signal is noise
        assert out.noisy_sample.abs().mean() > 0.1

    def test_linear_schedule(self):
        sched = DDPMScheduler(beta_schedule="linear")
        assert sched.betas[0] < sched.betas[-1]  # monotonically increasing

    def test_cosine_schedule_smooth(self):
        sched = DDPMScheduler(beta_schedule="cosine")
        diffs = sched.betas[1:] - sched.betas[:-1]
        assert (diffs >= -1e-6).all(), "Cosine schedule should be non-decreasing"


class TestDDIMScheduler:
    def test_set_timesteps(self):
        sched = DDIMScheduler(num_train_timesteps=1000)
        sched.set_timesteps(50)
        assert len(sched.timesteps) == 50

    def test_timesteps_decreasing(self):
        sched = DDIMScheduler(num_train_timesteps=1000)
        sched.set_timesteps(50)
        t = sched.timesteps
        assert (t[:-1] > t[1:]).all(), "Timesteps should be decreasing"


# ─────────────────────────────────────────────
#  EMA Tests
# ─────────────────────────────────────────────

class TestEMA:
    def test_shadow_params_init(self):
        model = build_unet("small")
        ema = EMAModel(model)
        assert len(ema.shadow_params) > 0

    def test_update_changes_shadow(self):
        model = build_unet("small")
        ema = EMAModel(model, decay=0.9)
        original = {k: v.clone() for k, v in ema.shadow_params.items()}

        # Modify model parameters
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.ones_like(p) * 10.0)

        ema.update(model)

        # Shadow should have changed
        changed = False
        for k in original:
            if not torch.allclose(original[k], ema.shadow_params[k], atol=0.1):
                changed = True
                break
        assert changed, "EMA shadow params should update after model weight change"

    def test_apply_and_restore(self):
        model = build_unet("small")
        ema = EMAModel(model)
        # Store original params
        orig = {n: p.clone() for n, p in model.named_parameters() if p.requires_grad}
        ema.apply_shadow(model)
        ema.restore(model)
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert torch.allclose(orig[name], param.data), f"Param {name} not restored"

    def test_warmup_decay(self):
        model = build_unet("small")
        ema = EMAModel(model, decay=0.9999, warmup_steps=100)
        # Early steps should have lower effective decay
        ema.step = 1
        early_decay = ema.get_decay()
        ema.step = 10000
        late_decay = ema.get_decay()
        assert late_decay >= early_decay


# ─────────────────────────────────────────────
#  Integration Tests
# ─────────────────────────────────────────────

class TestIntegration:
    def test_full_diffusion_step(self):
        """Test a complete forward + reverse step."""
        model = build_unet("small")
        sched = DDPMScheduler(num_train_timesteps=10)
        x = torch.randn(2, 3, 16, 16)
        t = torch.randint(0, 10, (2,))

        # Forward
        out = sched.add_noise(x, timesteps=t)
        noisy = out.noisy_sample

        # Predict
        with torch.no_grad():
            pred = model(noisy, t)

        assert pred.shape == x.shape

    def test_end_to_end_training_step(self):
        """Simulate one training iteration."""
        import torch.nn.functional as F

        model = build_unet("small")
        sched = DDPMScheduler(num_train_timesteps=100)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        x = torch.randn(2, 3, 16, 16)
        noise = torch.randn_like(x)
        t = torch.randint(0, 100, (2,))

        noisy = sched.add_noise(x, noise, t).noisy_sample
        pred = model(noisy, t)
        loss = F.mse_loss(pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert loss.item() > 0
        assert not torch.isnan(loss)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
