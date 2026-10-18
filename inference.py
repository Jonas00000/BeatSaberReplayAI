import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import bsor.Bsor as bsor_reader
import numpy as np
import torch

from features import (
    HISTORY_FRAMES,
    MODEL_INPUT_FEATURES,
    POSITION_INDICES,
    PREDICTION_FRAMES,
    TARGET_FPS,
    denormalize_positions,
    environment_features,
    load_map,
    normalize_positions,
    note_features,
    rotations_6d_to_quaternions,
)
from model import ReplayModel

DEFAULT_HEIGHT = 1.65
START_TIME = -1.0


def choose_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def directory_hash(path):
    digest = hashlib.sha1()
    for root, _, filenames in os.walk(path):
        for filename in sorted(filenames):
            with open(Path(root) / filename, "rb") as file:
                while chunk := file.read(1024 * 1024):
                    digest.update(chunk)
    return digest.hexdigest()


def load_statistics(path):
    with open(path, encoding="utf-8") as file:
        values = json.load(file)
    return np.asarray(values["mean"]), np.asarray(values["std"])


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint must contain model_state_dict")
    model = ReplayModel().to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model


def generate_motion(
    model,
    notes,
    walls,
    seed,
    mean,
    std,
    height=DEFAULT_HEIGHT,
    device="cpu",
):
    device = torch.device(device)
    raw_seed = np.zeros(MODEL_INPUT_FEATURES)
    raw_seed[:] = seed
    # Keep the captured height for warm-up; generated histories use the requested height.
    normalized_seed = normalize_positions(raw_seed, mean, std)
    history = torch.FloatTensor(np.tile(normalized_seed, (1, HISTORY_FRAMES, 1))).to(
        device
    )

    duration = notes[-1, 0] + 2.0 if len(notes) else 10.0
    current_time = START_TIME
    chunks = []

    with torch.no_grad():
        while current_time < duration:
            notes_input, density, upcoming_indices = note_features(notes, current_time)
            environment = environment_features(
                walls, current_time, density, inference_scaling=True
            )
            notes_tensor = torch.FloatTensor(notes_input).unsqueeze(0).to(device)
            environment_tensor = torch.FloatTensor(environment).unsqueeze(0).to(device)

            nearby_count = np.sum(notes[upcoming_indices[:10], 0] < current_time + 1.0)
            noise_level = min(0.002 + 0.02 * nearby_count, 0.2)
            noise = torch.zeros_like(history)
            noise[..., POSITION_INDICES] = (
                torch.randn_like(history[..., POSITION_INDICES]) * noise_level
            )
            noise[:, -5:, POSITION_INDICES] *= 0.5
            noise[:, -1:, POSITION_INDICES] = 0.0

            prediction = model(
                history + noise,
                notes_tensor,
                environment_tensor,
                steps=PREDICTION_FRAMES,
            )
            chunk = prediction.cpu().numpy()[0]
            chunks.append(chunk)

            height_column = np.full((PREDICTION_FRAMES, 1), height, dtype=np.float32)
            feedback = np.concatenate([chunk, height_column], axis=1)
            feedback = torch.FloatTensor(feedback).unsqueeze(0).to(device)
            history = feedback
            current_time += PREDICTION_FRAMES / TARGET_FPS

    return np.concatenate(chunks, axis=0)


def reconstruct_frames(predictions, mean, std):
    frames = []
    for index, prediction in enumerate(predictions):
        frame_time = START_TIME + index / TARGET_FPS
        if frame_time < 0:
            continue

        motion = denormalize_positions(prediction, mean, std)
        head_position = motion[0:3]
        head_rotation = rotations_6d_to_quaternions(motion[3:9])[0]
        left_position = motion[9:12] + head_position
        left_rotation = rotations_6d_to_quaternions(motion[12:18])[0]
        right_position = motion[18:21] + head_position
        right_rotation = rotations_6d_to_quaternions(motion[21:27])[0]
        frames.append(
            (
                frame_time,
                left_position,
                left_rotation,
                right_position,
                right_rotation,
                head_position,
                head_rotation,
            )
        )
    return frames


def _vr_object(position, rotation):
    value = bsor_reader.VRObject()
    value.x, value.y, value.z = map(float, position)
    value.x_rot, value.y_rot, value.z_rot, value.w_rot = map(float, rotation)
    return value


def _map_metadata(info, filename):
    environment = info.get("_environmentName", "DefaultEnvironment")
    for beatmap_set in info.get("_difficultyBeatmapSets", []):
        for difficulty in beatmap_set["_difficultyBeatmaps"]:
            if difficulty["_beatmapFilename"] == filename:
                mode = beatmap_set["_beatmapCharacteristicName"]
                environments = info.get("_environmentNames", [])
                index = difficulty.get("_environmentNameIdx")
                if index is not None and 0 <= index < len(environments):
                    environment = environments[index]
                return mode, difficulty["_difficulty"], environment
    return "Standard", Path(filename).stem.removesuffix("Standard"), environment


def write_replay(output_path, frames, map_folder, difficulty_filename, info, height):
    replay = bsor_reader.Bsor()
    replay.magic_number = 0x442D3D69
    replay.file_version = 1
    replay.info = bsor_reader.Info()
    replay.info.version = "2.0.0"
    replay.info.gameVersion = "1.39.0"
    replay.info.timestamp = str(int(time.time()))
    replay.info.playerId = "generated-replay"
    replay.info.playerName = "Generated Replay"
    replay.info.platform = ""
    replay.info.trackingSystem = "generated"
    replay.info.hmd = "generated"
    replay.info.controller = "generated"
    replay.info.songHash = directory_hash(map_folder)
    replay.info.songName = info.get("_songName", "Unknown Song")
    replay.info.mapper = info.get("_levelAuthorName", "Unknown Mapper")
    mode, difficulty, environment = _map_metadata(info, difficulty_filename)
    replay.info.difficulty = difficulty
    replay.info.score = 0
    replay.info.mode = mode
    replay.info.environment = environment
    replay.info.modifiers = ""
    replay.info.jumpDistance = 14.5
    replay.info.leftHanded = False
    replay.info.height = height
    replay.info.startTime = 0.0
    replay.info.failTime = 0.0
    replay.info.speed = 0.0

    replay.frames = []
    for (
        frame_time,
        left_pos,
        left_rot,
        right_pos,
        right_rot,
        head_pos,
        head_rot,
    ) in frames:
        frame = bsor_reader.Frame()
        frame.time = float(frame_time)
        frame.fps = TARGET_FPS
        frame.head = _vr_object(head_pos, head_rot)
        frame.left_hand = _vr_object(left_pos, left_rot)
        frame.right_hand = _vr_object(right_pos, right_rot)
        replay.frames.append(frame)

    replay.controller_offsets = []
    replay.notes = []
    replay.walls = []
    replay.heights = []
    replay.pauses = []
    replay.user_data = []

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as file:
        replay.write(file)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate an unscored Beat Saber replay"
    )
    parser.add_argument("--map", required=True, help="Folder containing Info.dat")
    parser.add_argument("--difficulty", required=True, help="Difficulty .dat filename")
    parser.add_argument("--checkpoint", default="checkpoints/pretrained/model.pth")
    parser.add_argument("--stats", default="checkpoints/pretrained/stats.json")
    parser.add_argument("--seed", default="checkpoints/pretrained/idle_seed.npy")
    parser.add_argument("--output", default="outputs/generated_replay.bsor")
    parser.add_argument("--height", type=float, default=DEFAULT_HEIGHT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--random-seed", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.random_seed is not None:
        torch.manual_seed(args.random_seed)

    device = choose_device(args.device)
    mean, std = load_statistics(args.stats)
    model = load_model(args.checkpoint, device)
    notes, walls, _, info = load_map(args.map, args.difficulty, note_order="inference")
    seed = np.load(args.seed)
    predictions = generate_motion(
        model, notes, walls, seed, mean, std, args.height, device
    )
    frames = reconstruct_frames(predictions, mean, std)
    write_replay(args.output, frames, args.map, args.difficulty, info, args.height)
    print(f"Wrote {len(frames)} frames to {args.output}")


if __name__ == "__main__":
    main()
