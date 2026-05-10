"""
Evaluation Script
Computes FID, IS, and LPIPS on generated samples.
Run: python scripts/evaluate.py --real_dir data/real --fake_dir outputs/samples
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from src.utils.metrics import EvaluationSuite
import json


def load_images_from_dir(path: str, image_size: int = 299, limit: int = 5000) -> torch.Tensor:
    """Load all images from a directory as a tensor."""
    tf = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
    ])
    ds = ImageFolder(root=path, transform=tf)
    dl = DataLoader(ds, batch_size=64, num_workers=4)
    images = []
    for batch, _ in dl:
        images.append(batch)
        if sum(len(b) for b in images) >= limit:
            break
    return torch.cat(images, dim=0)[:limit]


def main():
    parser = argparse.ArgumentParser(description="Evaluate generative model quality")
    parser.add_argument("--real_dir", type=str, required=True, help="Directory of real images")
    parser.add_argument("--fake_dir", type=str, required=True, help="Directory of generated images")
    parser.add_argument("--image_size", type=int, default=299)
    parser.add_argument("--num_images", type=int, default=5000)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default="outputs/eval_results.json")
    parser.add_argument("--skip_fid", action="store_true")
    parser.add_argument("--skip_is", action="store_true")
    parser.add_argument("--skip_lpips", action="store_true")
    args = parser.parse_args()

    print(f"\n📊 Evaluating generative model")
    print(f"   Real:   {args.real_dir}")
    print(f"   Fake:   {args.fake_dir}")
    print(f"   N:      {args.num_images}")
    print(f"   Device: {args.device}\n")

    real = load_images_from_dir(args.real_dir, args.image_size, args.num_images)
    fake = load_images_from_dir(args.fake_dir, args.image_size, args.num_images)

    suite = EvaluationSuite(args.device)
    results = suite.evaluate(
        real,
        fake,
        compute_fid=not args.skip_fid,
        compute_is=not args.skip_is,
        compute_lpips_score=not args.skip_lpips,
    )

    # Save results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved → {args.output}")


if __name__ == "__main__":
    main()
