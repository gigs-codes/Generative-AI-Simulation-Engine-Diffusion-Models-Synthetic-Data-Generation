"""
Training Script
Run with: python scripts/train.py --config configs/ddpm_cifar10.yaml
"""

import argparse
import sys
import yaml
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.trainer import DiffusionTrainer
from src.data.datasets import build_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Diffusion Model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
    parser.add_argument("--epochs", type=int, default=None, help="Override num_epochs")
    parser.add_argument("--device", type=str, default=None, help="Override device (cuda/cpu)")
    parser.add_argument("--run_name", type=str, default=None, help="WandB run name")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # CLI overrides
    if args.epochs:
        config["num_epochs"] = args.epochs
    if args.device:
        config["device"] = args.device
    if args.run_name:
        config["run_name"] = args.run_name

    print(f"\n🚀 Starting training with config: {args.config}")
    print(f"   Model:   {config.get('model_size', 'base')}")
    print(f"   Dataset: {config.get('dataset', 'cifar10')}")
    print(f"   Epochs:  {config.get('num_epochs', 100)}")
    print(f"   Device:  {config.get('device', 'cuda')}\n")

    # Build dataloaders
    train_dl, val_dl = build_dataloaders(config)

    # Build trainer
    trainer = DiffusionTrainer(config)

    # Resume if requested
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train(
        train_dataloader=train_dl,
        val_dataloader=val_dl,
        num_epochs=config.get("num_epochs", 100),
    )


if __name__ == "__main__":
    main()
