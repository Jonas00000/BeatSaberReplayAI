import argparse
import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import bsor.Bsor as bsor_reader
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp
from tqdm import tqdm

from features import (
    HISTORY_FRAMES,
    PREDICTION_FRAMES,
    TARGET_FPS,
    environment_features,
    load_map,
    note_features,
    quaternions_to_6d,
)

WINDOW_STRIDE = 5


class SkipReplay(Exception):
    pass


def resample_rotations(times, rotations, target_times):
    source = Rotation.from_quat(rotations)
    try:
        quaternions = Slerp(times, source)(target_times).as_quat()
    except ValueError:
        # Slerp cannot extrapolate when a recording starts after song time zero.
        quaternions = interp1d(
            times,
            rotations,
            axis=0,
            kind="linear",
            fill_value="extrapolate",
        )(target_times)
        quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)
    return quaternions_to_6d(quaternions)


def _check_replay(replay):
    if replay.info.speed > 0.01 and abs(replay.info.speed - 1.0) > 0.05:
        raise SkipReplay("Speed != 1.0")
    if len(replay.frames) < 100:
        raise SkipReplay("Too Short")
    if replay.frames[-1].time - 1.0 <= 0:
        raise SkipReplay("Invalid Duration")
    if len(replay.walls) > 4:
        raise SkipReplay("Wall Hit")

    misses = sum(note.event_type == 1 for note in replay.notes)
    bad_cuts = sum(note.event_type == 2 for note in replay.notes)
    if misses + bad_cuts > 30:
        raise SkipReplay("High Miss/Badcut")
    if sum(note.event_type == 3 for note in replay.notes) > 4:
        raise SkipReplay("Bomb Hits")


def _read_tracking(replay):
    valid_duration = replay.frames[-1].time - 1.0
    times = []
    positions = [[], [], []]
    rotations = [[], [], []]

    for frame in replay.frames:
        if frame.time > valid_duration:
            break
        times.append(frame.time)
        for index, pose in enumerate((frame.head, frame.left_hand, frame.right_hand)):
            positions[index].append([pose.x, pose.y, pose.z])
            rotations[index].append([pose.x_rot, pose.y_rot, pose.z_rot, pose.w_rot])

    arrays = [np.asarray(values) for values in (times, *positions, *rotations)]
    if np.any(np.abs(arrays[1]) > 20.0):
        raise SkipReplay("Head Outlier (Tracking Loss)")
    if np.any(np.abs(arrays[2]) > 20.0) or np.any(np.abs(arrays[3]) > 20.0):
        raise SkipReplay("Hand Outlier (Tracking Loss)")

    order = np.argsort(arrays[0])
    arrays = [array[order] for array in arrays]
    _, unique_indices = np.unique(arrays[0], return_index=True)
    arrays = [array[unique_indices] for array in arrays]
    if len(arrays[0]) < 50:
        raise SkipReplay("Low Frame Count")
    return arrays


def _resample_motion(replay):
    (
        times,
        head_positions,
        left_positions,
        right_positions,
        head_rotations,
        left_rotations,
        right_rotations,
    ) = _read_tracking(replay)
    target_times = np.arange(0, times[-1], 1 / TARGET_FPS)

    def resample_positions(values):
        return interp1d(times, values, axis=0, kind="linear", fill_value="extrapolate")(
            target_times
        )

    head = resample_positions(head_positions)
    left = resample_positions(left_positions) - head
    right = resample_positions(right_positions) - head
    player_height = (
        replay.info.height if replay.info.height > 0.1 else head_positions[0, 1]
    )
    height = np.full((len(target_times), 1), player_height)

    # Each pose is position + two rotation columns; hands are relative to the head.
    frames = np.concatenate(
        [
            head,
            resample_rotations(times, head_rotations, target_times),
            left,
            resample_rotations(times, left_rotations, target_times),
            right,
            resample_rotations(times, right_rotations, target_times),
            height,
        ],
        axis=1,
    )
    return target_times, frames


def process_replay(replay_path, map_path):
    with open(replay_path, "rb") as file:
        replay = bsor_reader.make_bsor(file)
    _check_replay(replay)
    target_times, frames = _resample_motion(replay)

    map_path = Path(map_path)
    notes, walls, _, _ = load_map(
        map_path.parent, map_path.name, note_order="preprocess"
    )
    histories, note_contexts, environments, targets = [], [], [], []
    sample_count = len(frames) - HISTORY_FRAMES - PREDICTION_FRAMES

    for start in range(0, sample_count, WINDOW_STRIDE):
        prediction_time = target_times[start + HISTORY_FRAMES]
        notes_input, density, _ = note_features(notes, prediction_time)
        environment = environment_features(walls, prediction_time, density)
        histories.append(frames[start : start + HISTORY_FRAMES])
        note_contexts.append(notes_input)
        environments.append(environment)
        targets.append(
            frames[start + HISTORY_FRAMES : start + HISTORY_FRAMES + PREDICTION_FRAMES]
        )

    histories = np.clip(histories, -65000.0, 65000.0)
    targets = np.clip(targets, -65000.0, 65000.0)
    return (
        np.asarray(histories, dtype=np.float16),
        np.asarray(note_contexts, dtype=np.float16),
        np.asarray(environments, dtype=np.float16),
        np.asarray(targets, dtype=np.float16),
    )


def _process_task(task):
    index, replay_path, map_path, output_dir = task
    output_path = Path(output_dir) / f"rep_{index}.npz"
    try:
        arrays = process_replay(replay_path, map_path)
        np.savez_compressed(
            output_path,
            xf=arrays[0],
            xn=arrays[1],
            xw=arrays[2],
            yf=arrays[3],
        )
        return "Saved"
    except SkipReplay as error:
        return f"Skipped: {error}"
    except (Exception, bsor_reader.BSException) as error:
        return f"Error: {error}"


def _load_index(path, replay_root=None):
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as file:
        entries = json.load(file)
    replay_base = Path(replay_root).resolve() if replay_root else path.parent
    return [
        (
            str((replay_base / entry["replay"]).resolve()),
            str((path.parent / entry["map"]).resolve()),
        )
        for entry in entries
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Convert replays into training windows"
    )
    parser.add_argument(
        "--index", action="append", required=True, help="May be supplied more than once"
    )
    parser.add_argument("--output", default="data/preprocessed")
    parser.add_argument("--replay-root", help="Optional base for relative replay paths")
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    args = parser.parse_args()

    entries = [
        entry for path in args.index for entry in _load_index(path, args.replay_root)
    ]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tasks = [
        (index, replay, map_path, str(output))
        for index, (replay, map_path) in enumerate(entries)
    ]

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(tqdm(executor.map(_process_task, tasks), total=len(tasks)))
    print("Processing complete:")
    for status, count in Counter(results).items():
        print(f"  {status}: {count}")


if __name__ == "__main__":
    main()
