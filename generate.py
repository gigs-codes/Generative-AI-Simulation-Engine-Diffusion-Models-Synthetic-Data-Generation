"""
Generation Script
Run with: python scripts/generate.py --config configs/ddpm_cifar10.yaml --num_samples 100
"""

import argparse
import sys
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.pipelines.ddpm_pipeline import DDPMPipeline, DDIMPipeline
from src.models.noise_scheduler import DDPMScheduler, DDIMScheduler
from src.models.unet import build_unet
from src.pipelines.synthetic_data_pipeline import SyntheticDataGenerator, GenerationConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Synthetic Samples")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--output", type=str, default="outputs/samples")
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--sampler", type=str, choices=["ddpm", "ddim"], default="ddim")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--class_label", type=int, default=None)
    parser.add_argument("--save_grid", action="store_true", help="Save a grid image")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = args.device or config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n🎨 Generating {args.num_samples} samples")
    print(f"   Sampler: {args.sampler.upper()} ({args.num_steps} steps)")
    print(f"   Device:  {device}")
    print(f"   Output:  {args.output}\n")

    # Build model
    unet = build_unet(
        size=config.get("model_size", "base"),
        in_channels=config.get("in_channels", 3),
        out_channels=config.get("in_channels", 3),
        num_classes=config.get("num_classes", 0),
    )

    # Build pipeline
    if args.sampler == "ddim":
        scheduler = DDIMScheduler(
            num_train_timesteps=config.get("num_timesteps", 1000),
            beta_schedule=config.get("beta_schedule", "cosine"),
        )
        pipeline = DDIMPipeline(unet, scheduler, config.get("image_size", 32), device)
    else:
        scheduler = DDPMScheduler(
            num_train_timesteps=config.get("num_timesteps", 1000),
            beta_schedule=config.get("beta_schedule", "cosine"),
        )
        pipeline = DDPMPipeline(unet, scheduler, config.get("image_size", 32), device)

    # Load checkpoint
    if args.checkpoint:
        pipeline.load_checkpoint(args.checkpoint)

    # Build generator
    gen_config = GenerationConfig(
        num_samples=args.num_samples,
        batch_size=config.get("batch_size", 32),
        image_size=config.get("image_size", 32),
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_steps,
        seed=args.seed,
        output_dir=args.output,
    )
    generator = SyntheticDataGenerator(pipeline, gen_config)

    # Generate
    class_labels = [args.class_label] * args.num_samples if args.class_label is not None else None
    results = generator.generate_dataset(
        class_labels=class_labels,
        num_classes=config.get("num_classes") if not class_labels else None,
        balanced=True,
    )

    # Save grid
    if args.save_grid:
        import torchvision.utils as vutils
        from PIL import Image
        import numpy as np

        images = results["images"][:64]  # max 8x8 grid
        grid = vutils.make_grid(images, nrow=8, normalize=False, padding=2)
        grid_np = (grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        grid_path = Path(args.output) / "grid.png"
        Image.fromarray(grid_np).save(grid_path)
        print(f"  ✓ Grid saved → {grid_path}")


if __name__ == "__main__":
    main()
