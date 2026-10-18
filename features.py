import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

TARGET_FPS = 60
HISTORY_FRAMES = 30
PREDICTION_FRAMES = 30
NOTE_SLOTS = 30
NOTE_FEATURES = 21
MODEL_INPUT_FEATURES = 28
# Head position, left hand relative to head, right hand relative to head.
POSITION_INDICES = (0, 1, 2, 9, 10, 11, 18, 19, 20)

_DIRECTION_SORT = {
    0: (-1, -1),
    1: (-1, 1),
    2: (1, 1),
    3: (1, 1),
    4: (-1, -1),
    5: (1, -1),
    6: (-1, 1),
    7: (1, 1),
    8: (-1, 1),
    9: (-1, 1),
}


def beat_to_time(beat, initial_bpm, bpm_changes):
    if not bpm_changes:
        return beat * (60.0 / initial_bpm)

    events = sorted(bpm_changes, key=lambda event: event["b"])
    if events[0]["b"] > 0:
        events.insert(0, {"b": 0, "m": initial_bpm})

    current_time = 0.0
    for index, event in enumerate(events):
        next_event = events[index + 1] if index + 1 < len(events) else None
        if next_event is None or beat < next_event["b"]:
            current_time += (beat - event["b"]) * (60.0 / event["m"])
            return current_time
        current_time += (next_event["b"] - event["b"]) * (60.0 / event["m"])
    return current_time


def _preprocess_note_key(note):
    time, line, layer, color, direction = note
    x_direction, y_direction = _DIRECTION_SORT.get(int(direction), (-1, 1))
    if int(color) == 0:
        x_direction *= -1
    layer_key = -layer if y_direction == 1 else layer
    line_key = -line if x_direction == 1 else line
    return time, color, int(direction), layer_key, line_key


def load_map(map_folder, difficulty_filename, *, note_order="inference"):
    """Read v2/v3 notes and walls, preserving the ordering used in each stage."""
    map_folder = Path(map_folder)
    with (map_folder / "Info.dat").open(encoding="utf-8") as file:
        info = json.load(file)
    with (map_folder / difficulty_filename).open(encoding="utf-8") as file:
        difficulty = json.load(file)

    bpm = info["_beatsPerMinute"]
    version = difficulty.get("version", difficulty.get("_version", "2.0.0"))
    if not version.startswith(("2", "3")):
        raise ValueError(f"Unsupported map version: {version}")
    is_v3 = version.startswith("3")
    if is_v3:
        bpm_changes = difficulty.get("bpmEvents", [])
    else:
        bpm_changes = [
            {"b": event["_time"], "m": event["_floatValue"]}
            for event in difficulty.get("_events", [])
            if event["_type"] == 100
        ]

    def seconds(beat):
        return beat_to_time(beat, bpm, bpm_changes)

    notes = []
    if is_v3:
        for note in difficulty.get("colorNotes", []):
            notes.append(
                [seconds(note["b"]), note["x"], note["y"], note["c"], note["d"]]
            )
        for chain in difficulty.get("burstSliders", []):
            notes.append(
                [seconds(chain["tb"]), chain["tx"], chain["ty"], chain["c"], 8]
            )
        for bomb in difficulty.get("bombNotes", []):
            notes.append([seconds(bomb["b"]), bomb["x"], bomb["y"], 2, 9])
    else:
        for note in difficulty.get("_notes", []):
            note_type = note.get("_type", 0)
            if note_type in (0, 1):
                notes.append(
                    [
                        seconds(note["_time"]),
                        note["_lineIndex"],
                        note["_lineLayer"],
                        note_type,
                        note["_cutDirection"],
                    ]
                )
            elif note_type == 3:
                notes.append(
                    [
                        seconds(note["_time"]),
                        note["_lineIndex"],
                        note["_lineLayer"],
                        2,
                        9,
                    ]
                )

    if note_order == "preprocess":
        notes.sort(key=_preprocess_note_key)
    elif note_order == "inference":
        notes.sort(key=lambda note: (note[0], note[1], note[2]))
    else:
        raise ValueError(f"Unknown note ordering: {note_order}")

    walls = []
    if is_v3:
        for wall in difficulty.get("obstacles", []):
            start = seconds(wall["b"])
            end = seconds(wall["b"] + wall["d"])
            walls.append(
                [start, end - start, wall["x"], wall["y"], wall["w"], wall["h"]]
            )
    else:
        for wall in difficulty.get("_obstacles", []):
            start = seconds(wall["_time"])
            end = seconds(wall["_time"] + wall["_duration"])
            is_full_height = wall["_type"] == 0
            walls.append(
                [
                    start,
                    end - start,
                    wall["_lineIndex"],
                    0 if is_full_height else 2,
                    wall["_width"],
                    5 if is_full_height else 3,
                ]
            )

    return (
        np.asarray(notes, dtype=np.float64).reshape(-1, 5),
        np.asarray(walls, dtype=np.float64).reshape(-1, 6),
        bpm,
        info,
    )


def encode_note(note):
    time_offset, line, layer, note_type, direction = note
    line = int(np.clip(line, 0, 3))
    layer = int(np.clip(layer, 0, 2))
    note_type = int(np.clip(note_type, 0, 2))
    direction = int(np.clip(direction, 0, 9))
    return np.concatenate(
        [
            [time_offset],
            np.eye(4)[line],
            np.eye(3)[layer],
            np.eye(3)[note_type],
            np.eye(10)[direction],
        ]
    )


def note_density(notes):
    """Count distinct time/color pairs per second, scaled by 1/10 for the model."""
    colored = notes[(notes[:, 3] == 0) | (notes[:, 3] == 1)]
    if len(colored) == 0:
        return 0.0
    unique_swings = np.unique(np.c_[np.round(colored[:, 0], 3), colored[:, 3]], axis=0)
    return (len(unique_swings) / 2.0) / 10.0


def note_features(notes, prediction_time):
    upcoming = np.where(
        (notes[:, 0] > prediction_time) & (notes[:, 0] <= prediction_time + 2.0)
    )[0]
    if len(upcoming) == 0:
        return np.zeros((NOTE_SLOTS, NOTE_FEATURES)), 0.0, upcoming

    all_upcoming = notes[upcoming]
    density = note_density(all_upcoming)
    selected = all_upcoming[:NOTE_SLOTS].copy()
    selected[:, 0] -= prediction_time
    encoded = np.asarray([encode_note(note) for note in selected])
    if len(encoded) < NOTE_SLOTS:
        encoded = np.vstack(
            [encoded, np.zeros((NOTE_SLOTS - len(encoded), NOTE_FEATURES))]
        )
    return encoded, density, upcoming


def wall_features(walls, start_time, duration, fps=TARGET_FPS):
    relative_times = np.arange(0, duration, 1 / fps)
    times = start_time + relative_times
    features = np.zeros((len(times), 4), dtype=np.float16)
    if len(walls) == 0:
        return features

    lookahead = 0.6
    window_end = start_time + duration
    relevant = walls[
        (walls[:, 0] < window_end + lookahead)
        & ((walls[:, 0] + walls[:, 1]) > start_time)
    ]
    zones = ((1, 1), (2, 1), (1, 2), (2, 2))
    for wall in relevant:
        wall_time, wall_duration, x, y, width, height = wall
        active = (times >= wall_time) & (times < wall_time + wall_duration)
        approaching = (times >= wall_time - lookahead) & (times < wall_time)
        covered_zones = [
            index
            for index, (zone_x, zone_y) in enumerate(zones)
            if x <= zone_x < x + width and y <= zone_y < y + height
        ]
        for index in covered_zones:
            features[active, index] = 1.0
            if np.any(approaching):
                ramp = 1.0 - ((wall_time - times[approaching]) / lookahead)
                features[approaching, index] = np.maximum(
                    features[approaching, index], ramp
                )
    return features


def environment_features(walls, prediction_time, density, *, inference_scaling=False):
    features = wall_features(walls, prediction_time, PREDICTION_FRAMES / TARGET_FPS)
    # Training stores float16; inference retains density precision until tensor conversion.
    density_dtype = np.float64 if inference_scaling else np.float16
    features = np.hstack(
        [
            features,
            np.full((PREDICTION_FRAMES, 1), density, dtype=density_dtype),
        ]
    )
    if inference_scaling:
        features[:, :4] *= 2.0
    return features


def quaternions_to_6d(quaternions):
    """Store the first two rotation-matrix columns, each as three consecutive values."""
    matrices = Rotation.from_quat(quaternions).as_matrix()
    return matrices[:, :, :2].transpose(0, 2, 1).reshape(-1, 6)


def rotations_6d_to_quaternions(rotations):
    # Gram-Schmidt restores perpendicular axes before quaternion conversion.
    rotations = np.atleast_2d(rotations)
    first = rotations[:, :3]
    second = rotations[:, 3:6]
    first = first / (np.linalg.norm(first, axis=1, keepdims=True) + 1e-8)
    second = second - np.sum(first * second, axis=1, keepdims=True) * first
    second = second / (np.linalg.norm(second, axis=1, keepdims=True) + 1e-8)
    third = np.cross(first, second)
    return Rotation.from_matrix(np.stack([first, second, third], axis=2)).as_quat()


def normalize_positions(frames, mean, std):
    result = frames.copy()
    for stats_index, frame_index in enumerate(POSITION_INDICES):
        result[..., frame_index] = (result[..., frame_index] - mean[stats_index]) / std[
            stats_index
        ]
    return result


def denormalize_positions(frames, mean, std):
    result = frames.copy()
    for stats_index, frame_index in enumerate(POSITION_INDICES):
        result[..., frame_index] = (
            result[..., frame_index] * std[stats_index] + mean[stats_index]
        )
    return result
