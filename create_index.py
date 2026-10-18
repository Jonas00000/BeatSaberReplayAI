import argparse
import json
from pathlib import Path

import bsor.Bsor as bsor_reader
import numpy as np
from tqdm import tqdm


def _sorted_notes(notes):
    notes = np.asarray(notes).reshape(-1, 5)
    order = np.lexsort(
        (notes[:, 4], notes[:, 3], notes[:, 2], notes[:, 1], notes[:, 0])
    )
    return notes[order]


def _map_notes(path):
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    version = data.get("version", data.get("_version", "2.0.0"))
    if version.startswith("3"):
        notes = [
            [note["b"], note["x"], note["y"], note["c"], note["d"]]
            for note in data.get("colorNotes", [])
        ]
    elif version.startswith("2"):
        notes = [
            [
                note["_time"],
                note["_lineIndex"],
                note["_lineLayer"],
                note["_type"],
                note["_cutDirection"],
            ]
            for note in data.get("_notes", [])
            if note["_type"] in (0, 1)
        ]
    else:
        raise ValueError(f"Unsupported map version: {version}")
    return _sorted_notes(notes)


def create_index(replay_dir, map_dirs, output):
    candidates = []
    seen = set()
    for folder in map_dirs:
        for info_path in sorted(Path(folder).rglob("Info.dat")):
            try:
                with info_path.open(encoding="utf-8") as file:
                    info = json.load(file)
                for beatmap_set in info.get("_difficultyBeatmapSets", []):
                    mode = beatmap_set["_beatmapCharacteristicName"]
                    for difficulty in beatmap_set["_difficultyBeatmaps"]:
                        path = (
                            info_path.parent / difficulty["_beatmapFilename"]
                        ).resolve()
                        if path not in seen:
                            candidates.append(
                                (
                                    path,
                                    mode,
                                    difficulty["_difficulty"],
                                    _map_notes(path),
                                )
                            )
                            seen.add(path)
            except Exception as error:
                print(f"Skipping {info_path}: {error}")
    if not candidates:
        raise ValueError("No supported map difficulties found")

    output = Path(output).resolve()
    result = []
    for replay_path in tqdm(sorted(Path(replay_dir).rglob("*.bsor")), unit="replay"):
        try:
            with replay_path.open("rb") as file:
                replay = bsor_reader.make_bsor(file)
            if replay.info.leftHanded:
                continue
            replay_notes = _sorted_notes(
                [
                    [
                        note.spawn_time,
                        note.lineIndex,
                        note.noteLineLayer,
                        note.colorType,
                        note.cutDirection,
                    ]
                    for note in replay.notes
                    if note.event_type in (0, 1, 2) and note.scoringType in (3, 4, 5, 6)
                ]
            )
            if len(replay_notes) < 5:
                continue
            matches = [
                path
                for path, mode, difficulty, notes in candidates
                if mode == replay.info.mode
                and difficulty == replay.info.difficulty
                and len(notes) >= len(replay_notes)
                and np.array_equal(replay_notes[:, 1:], notes[: len(replay_notes), 1:])
            ]
            if len(matches) != 1:
                print(
                    f"Skipping {replay_path.name}: {len(matches)} matching difficulties"
                )
                continue
            result.append(
                {
                    "replay": replay_path.resolve().as_posix(),
                    "map": matches[0].as_posix(),
                }
            )
        except (Exception, bsor_reader.BSException) as error:
            print(f"Skipping {replay_path.name}: {error}")

    if not result:
        raise ValueError("No replay/map pairs found")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(result)} replay/map pairs to {output}")


def main():
    parser = argparse.ArgumentParser(
        description="Match local replays to map difficulties"
    )
    parser.add_argument("--replays", default="data/replays")
    parser.add_argument("--maps", nargs="+", default=["data/maps"])
    parser.add_argument("--output", default="data/index.json")
    args = parser.parse_args()
    create_index(args.replays, args.maps, args.output)


if __name__ == "__main__":
    main()
