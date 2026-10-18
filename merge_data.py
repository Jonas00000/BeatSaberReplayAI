import argparse
import glob
import json
import os
import random
from pathlib import Path

import numpy as np
from tqdm import tqdm

from features import POSITION_INDICES


def _position_statistics(frames):
    head = frames[:, 0:3]
    left = frames[:, 9:12]
    mirrored_right = frames[:, 18:21].copy()
    mirrored_right[:, 0] *= -1

    head_mean = np.mean(head, axis=0)
    head_std = np.std(head, axis=0)
    head_std[head_std < 1e-5] = 1.0

    hands = np.concatenate([left, mirrored_right], axis=0)
    hand_mean = np.mean(hands, axis=0)
    hand_std = np.std(hands, axis=0)
    hand_std[hand_std < 1e-5] = 1.0

    right_mean = hand_mean.copy()
    right_mean[0] *= -1
    mean = np.concatenate([head_mean, hand_mean, right_mean])
    std = np.concatenate([head_std, hand_std, hand_std])
    return mean, std


def merge_dataset(source_dir, output_dir, seed=None):
    files = glob.glob(os.path.join(source_dir, "*.npz"))
    if not files:
        raise ValueError(f"No .npz files found in {source_dir}")

    shapes = None
    valid_files = []
    total_samples = 0
    for path in tqdm(files, desc="Scanning"):
        try:
            with np.load(path) as data:
                history = data["xf"]
                count = len(history)
                if count == 0:
                    continue
                if shapes is None:
                    shapes = (
                        history.shape[1:],
                        *(data[key].shape[1:] for key in ("xn", "xw", "yf")),
                    )
            total_samples += count
            valid_files.append(path)
        except Exception as error:
            print(f"Skipping corrupt file {path}: {error}")
    if total_samples == 0:
        raise ValueError("The processed files contain no training windows")

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    random.shuffle(valid_files)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    names = ("all_xf.npy", "all_xn.npy", "all_xw.npy", "all_yf.npy")
    arrays = [
        np.lib.format.open_memmap(
            output_dir / name,
            mode="w+",
            dtype=np.float16,
            shape=(total_samples, *shape),
        )
        for name, shape in zip(names, shapes)
    ]

    offset = 0
    for file_index, path in enumerate(tqdm(valid_files, desc="Merging")):
        with np.load(path) as data:
            count = data["xf"].shape[0]
            for destination, key in zip(arrays, ("xf", "xn", "xw", "yf")):
                destination[offset : offset + count] = data[key]
            offset += count
        if file_index % 50 == 0:
            for array in arrays:
                array.flush()
    for array in arrays:
        array.flush()

    sampled_indices = np.random.choice(
        total_samples, min(100_000, total_samples), replace=False
    )
    sampled_frames = arrays[0][sampled_indices, 0, :].astype(np.float32)
    mean, std = _position_statistics(sampled_frames)

    for block_index, start in enumerate(
        tqdm(range(0, total_samples, 50_000), desc="Normalizing")
    ):
        end = min(start + 50_000, total_samples)
        histories = arrays[0][start:end].astype(np.float32)
        targets = arrays[3][start:end].astype(np.float32)
        for stats_index, position_index in enumerate(POSITION_INDICES):
            histories[:, :, position_index] = (
                histories[:, :, position_index] - mean[stats_index]
            ) / std[stats_index]
            targets[:, :, position_index] = (
                targets[:, :, position_index] - mean[stats_index]
            ) / std[stats_index]
        arrays[0][start:end] = histories.astype(np.float16)
        arrays[3][start:end] = targets.astype(np.float16)
        if block_index % 10 == 0:
            arrays[0].flush()
            arrays[3].flush()
    arrays[0].flush()
    arrays[3].flush()
    with (output_dir / "stats.json").open("w", encoding="utf-8") as file:
        json.dump({"mean": mean.tolist(), "std": std.tolist()}, file)
    print(f"Merged {total_samples} windows into {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge and normalize processed replay windows"
    )
    parser.add_argument("--input", default="data/preprocessed")
    parser.add_argument("--output", default="data/merged")
    parser.add_argument(
        "--seed",
        type=int,
        help="Optional reproducible file and statistics sampling seed",
    )
    args = parser.parse_args()
    merge_dataset(args.input, args.output, args.seed)


if __name__ == "__main__":
    main()
