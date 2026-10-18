import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Extract a starting pose from a processed replay"
    )
    parser.add_argument("replay", help="A .npz file produced by preprocess.py")
    parser.add_argument("--output", default="checkpoints/training/idle_seed.npy")
    args = parser.parse_args()

    with np.load(args.replay) as data:
        seed = data["xf"][0, 0].copy()
    if seed.shape != (28,) or not np.isfinite(seed).all():
        raise ValueError("Expected a finite 28-value pose")
    seed[0] = 0.0
    seed[2] = 0.0
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, seed)
    print(f"Saved starting pose to {output}")


if __name__ == "__main__":
    main()
