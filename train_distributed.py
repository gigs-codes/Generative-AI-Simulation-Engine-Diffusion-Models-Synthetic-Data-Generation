"""
Distributed Training Script
Run with: accelerate launch --num_processes 4 scripts/train_distributed.py --config configs/ldm_large.yaml
"""

import argparse
import sys
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.training.trainer import DiffusionTrainer
from src.data.datasets import build_dataloaders


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Force accelerate usage for distributed
    config["use_accelerate"] = True

    train_dl, val_dl = build_dataloaders(config)
    trainer = DiffusionTrainer(config)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train(
        train_dataloader=train_dl,
        val_dataloader=val_dl,
        num_epochs=config.get("num_epochs", 200),
    )


if __name__ == "__main__":
    main()
