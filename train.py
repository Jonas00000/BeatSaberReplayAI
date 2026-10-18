import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from features import POSITION_INDICES
from model import ReplayModel

DEFAULT_BATCH_SIZE = 2048
DEFAULT_EPOCHS = 200
DEFAULT_LEARNING_RATE = 0.0005
DEFAULT_WEIGHT_DECAY = 1e-5
DEFAULT_SAMPLES_PER_EPOCH = 2_000_000
TARGET_LENGTH = 30
WALL_MIRROR_ORDER = (1, 0, 3, 2)
LINE_MIRROR_ORDER = (3, 2, 1, 0)
COLOR_MIRROR_ORDER = (1, 0, 2)
DIRECTION_MIRROR_ORDER = (0, 1, 3, 2, 5, 4, 7, 6, 8, 9)


class ReplayLoss(nn.Module):
    def __init__(
        self,
        mean=None,
        std=None,
        frame_duration=1.0 / 60.0,
        speed_scale=3.0,
        speed_cap=5.0,
    ):
        super().__init__()
        self.position_indices = POSITION_INDICES
        if mean is not None:
            self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
            self.register_buffer("std", torch.tensor(std, dtype=torch.float32))
        else:
            self.mean = None
            self.std = None

        self.frame_duration = frame_duration
        self.speed_scale = float(speed_scale)
        self.speed_cap = float(speed_cap)
        self.velocity_weight = 0.5
        self.acceleration_weight = 0.1
        self.orthogonality_weight = 0.1
        self.rotation_weight = 0.5
        self.tip_weight = 0.5
        self.direction_weight = 1.0
        self.smooth_window = 3
        self.l1 = nn.L1Loss(reduction="none")
        self.huber = nn.SmoothL1Loss(reduction="none")

    def _denormalize_positions(self, values):
        positions = values[..., self.position_indices]
        if self.mean is not None:
            positions = positions * self.std.view(1, 1, -1) + self.mean.view(1, 1, -1)
        return positions

    def _smooth(self, values):
        if self.smooth_window <= 1:
            return values
        padding = (self.smooth_window - 1) // 2
        padded = functional.pad(
            values.permute(0, 2, 1), (padding, padding), mode="replicate"
        )
        kernel = torch.ones(1, 1, self.smooth_window, device=values.device) / float(
            self.smooth_window
        )
        return functional.conv1d(padded, kernel).permute(0, 2, 1)

    @staticmethod
    def _orthogonality(rotation):
        first = functional.normalize(rotation[:, 0:3], dim=1)
        second = functional.normalize(rotation[:, 3:6], dim=1)
        return torch.mean(torch.sum(first * second, dim=1) ** 2)

    @staticmethod
    def _saber_tip(position, rotation):
        first = functional.normalize(rotation[..., 0:3], dim=-1)
        second = functional.normalize(rotation[..., 3:6], dim=-1)
        third = torch.cross(first, second, dim=-1)
        # Approximate a one-metre saber extending along the controller's negative Z axis.
        return position - third

    def forward(self, prediction, target):
        sequence_length = prediction.shape[1]
        temporal_weights = torch.linspace(
            1.0, 0.3, sequence_length, device=prediction.device
        ).view(1, sequence_length, 1)

        target_positions = self._denormalize_positions(target)
        predicted_positions = self._denormalize_positions(prediction)
        target_head = target_positions[..., 0:3]
        predicted_head = predicted_positions[..., 0:3]
        target_left = target_head + target_positions[..., 3:6]
        target_right = target_head + target_positions[..., 6:9]
        predicted_left = predicted_head + predicted_positions[..., 3:6]
        predicted_right = predicted_head + predicted_positions[..., 6:9]

        target_left_displacement = target_left[:, 1:] - target_left[:, :-1]
        target_right_displacement = target_right[:, 1:] - target_right[:, :-1]
        predicted_left_displacement = predicted_left[:, 1:] - predicted_left[:, :-1]
        predicted_right_displacement = predicted_right[:, 1:] - predicted_right[:, :-1]

        fps = 1.0 / self.frame_duration
        # The training loss wraps the first transition to fill the final frame weight.
        left_speed = torch.cat(
            [torch.norm(target_left_displacement * fps, p=2, dim=2, keepdim=True)] * 2,
            dim=1,
        )[:, :sequence_length, :]
        right_speed = torch.cat(
            [torch.norm(target_right_displacement * fps, p=2, dim=2, keepdim=True)] * 2,
            dim=1,
        )[:, :sequence_length, :]
        left_weight = 1.0 + torch.clamp(
            self._smooth(left_speed) * self.speed_scale, max=self.speed_cap
        )
        right_weight = 1.0 + torch.clamp(
            self._smooth(right_speed) * self.speed_scale, max=self.speed_cap
        )

        base_loss = (self.l1(prediction, target) * temporal_weights).mean()
        absolute_position_loss = (
            self.l1(predicted_left, target_left) * left_weight * temporal_weights
        ).mean() + (
            self.l1(predicted_right, target_right) * right_weight * temporal_weights
        ).mean()
        base_loss += absolute_position_loss

        velocity_loss = (
            torch.abs(predicted_left_displacement - target_left_displacement)
            * left_weight[:, :-1, :]
        ).mean() + (
            torch.abs(predicted_right_displacement - target_right_displacement)
            * right_weight[:, :-1, :]
        ).mean()

        acceleration_loss = functional.l1_loss(
            predicted_left_displacement[:, 1:] - predicted_left_displacement[:, :-1],
            target_left_displacement[:, 1:] - target_left_displacement[:, :-1],
        ) + functional.l1_loss(
            predicted_right_displacement[:, 1:] - predicted_right_displacement[:, :-1],
            target_right_displacement[:, 1:] - target_right_displacement[:, :-1],
        )

        left_similarity = functional.cosine_similarity(
            predicted_left_displacement,
            target_left_displacement,
            dim=2,
            eps=1e-6,
        ).unsqueeze(-1)
        right_similarity = functional.cosine_similarity(
            predicted_right_displacement,
            target_right_displacement,
            dim=2,
            eps=1e-6,
        ).unsqueeze(-1)
        left_active = (left_speed[:, :-1, :] > 2.0).float()
        right_active = (right_speed[:, :-1, :] > 2.0).float()
        direction_loss = (
            (1.0 - left_similarity) * left_weight[:, :-1, :] * left_active
        ).mean() + (
            (1.0 - right_similarity) * right_weight[:, :-1, :] * right_active
        ).mean()

        flat_prediction = prediction.view(-1, prediction.shape[-1])
        orthogonality_loss = (
            self._orthogonality(flat_prediction[:, 3:9])
            + self._orthogonality(flat_prediction[:, 12:18])
            + self._orthogonality(flat_prediction[:, 21:27])
        )

        target_left_rotation = target[..., 12:18]
        predicted_left_rotation = prediction[..., 12:18]
        target_right_rotation = target[..., 21:27]
        predicted_right_rotation = prediction[..., 21:27]
        rotation_velocity_loss = (
            self.huber(
                predicted_left_rotation[:, 1:] - predicted_left_rotation[:, :-1],
                target_left_rotation[:, 1:] - target_left_rotation[:, :-1],
            )
            * left_weight[:, :-1]
        ).mean() + (
            self.huber(
                predicted_right_rotation[:, 1:] - predicted_right_rotation[:, :-1],
                target_right_rotation[:, 1:] - target_right_rotation[:, :-1],
            )
            * right_weight[:, :-1]
        ).mean()

        predicted_left_tip = self._saber_tip(predicted_left, predicted_left_rotation)
        target_left_tip = self._saber_tip(target_left, target_left_rotation)
        predicted_right_tip = self._saber_tip(predicted_right, predicted_right_rotation)
        target_right_tip = self._saber_tip(target_right, target_right_rotation)
        tip_loss = (
            self.huber(predicted_left_tip, target_left_tip)
            * left_weight
            * temporal_weights
        ).mean() + (
            self.huber(predicted_right_tip, target_right_tip)
            * right_weight
            * temporal_weights
        ).mean()

        return (
            base_loss
            + self.velocity_weight * velocity_loss
            + self.direction_weight * direction_loss
            + self.acceleration_weight * acceleration_loss
            + self.orthogonality_weight * orthogonality_loss
            + self.rotation_weight * rotation_velocity_loss
            + self.tip_weight * tip_loss
        )


def _mirror_pose(block):
    block[:, 0] *= -1
    block[:, 4:7] *= -1
    return block


def mirror_example(history, notes, environment, target):
    """Mirror a stored example in place, using the training coordinate convention."""
    for motion in (history, target):
        motion[:, :9] = _mirror_pose(motion[:, :9].clone())
        left = _mirror_pose(motion[:, 9:18].clone())
        right = _mirror_pose(motion[:, 18:27].clone())
        motion[:, 9:18], motion[:, 18:27] = right, left

    environment[:, :4] = environment[:, :4][:, WALL_MIRROR_ORDER]
    notes[:, 1:5] = notes[:, 1:5][:, LINE_MIRROR_ORDER]
    notes[:, 8:11] = notes[:, 8:11][:, COLOR_MIRROR_ORDER]
    notes[:, 11:21] = notes[:, 11:21][:, DIRECTION_MIRROR_ORDER]


def sort_note_context(notes):
    """Sort by time, lane, then layer, with padding at the end."""
    padding = notes[:, 1:5].sum(dim=1) == 0
    times = notes[:, 0].clone()
    times[padding] = 99999.0

    order = torch.argsort(torch.argmax(notes[:, 5:8], dim=1), stable=True)
    notes, times = notes[order], times[order]
    order = torch.argsort(torch.argmax(notes[:, 1:5], dim=1), stable=True)
    notes, times = notes[order], times[order]
    return notes[torch.argsort(times, stable=True)]


class ReplayDataset(IterableDataset):
    def __init__(
        self,
        data_dir,
        mode="train",
        val_split=0.05,
        chunk_size=32768,
        samples_per_epoch=DEFAULT_SAMPLES_PER_EPOCH,
    ):
        if mode not in ("train", "val"):
            raise ValueError("Dataset mode must be 'train' or 'val'")
        if chunk_size < 1 or samples_per_epoch < 1 or not 0 < val_split < 1:
            raise ValueError(
                "Chunk size and sample count must be positive, with 0 < val_split < 1"
            )
        self.mode = mode
        self.chunk_size = chunk_size
        self.samples_per_epoch = samples_per_epoch
        folder = Path(data_dir)
        self.paths = [
            folder / name
            for name in ("all_xf.npy", "all_xn.npy", "all_xw.npy", "all_yf.npy")
        ]
        total = np.load(self.paths[0], mmap_mode="r").shape[0]
        validation_size = int(total * val_split)
        if mode == "train":
            self.start, self.end = 0, total - validation_size
        else:
            self.start, self.end = total - validation_size, total
        self.val_split = val_split
        if self.end <= self.start or len(self) == 0:
            raise ValueError(
                f"No {mode} examples available; increase the dataset or sample budget"
            )

    def __len__(self):
        if self.mode == "train":
            return self.samples_per_epoch
        return int(self.samples_per_epoch * self.val_split)

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        worker_count = worker.num_workers if worker else 1
        worker_samples = len(self) // worker_count
        if worker_samples == 0:
            raise ValueError("Sample budget is smaller than the worker count")
        arrays = [np.load(path, mmap_mode="r") for path in self.paths]

        yielded = 0
        while yielded < worker_samples:
            maximum_start = max(self.start, self.end - self.chunk_size)
            chunk_start = np.random.randint(self.start, maximum_start + 1)
            size = min(self.chunk_size, self.end - chunk_start)
            chunks = [
                array[chunk_start : chunk_start + size].copy() for array in arrays
            ]
            indices = np.arange(size)
            np.random.shuffle(indices)

            for index in indices:
                if yielded >= worker_samples:
                    break
                history, notes, environment, target = [
                    torch.from_numpy(chunk[index]).float() for chunk in chunks
                ]
                if self.mode == "train" and random.random() > 0.5:
                    mirror_example(history, notes, environment, target)
                notes = sort_note_context(notes)
                yield history, notes, environment, target
                yielded += 1


def _loader(dataset, batch_size, workers):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )


def train(args):
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    data_dir = Path(args.data)
    with (data_dir / "stats.json").open(encoding="utf-8") as file:
        stats = json.load(file)

    train_dataset = ReplayDataset(
        data_dir, mode="train", samples_per_epoch=args.samples_per_epoch
    )
    validation_dataset = ReplayDataset(
        data_dir, mode="val", samples_per_epoch=args.samples_per_epoch
    )
    train_loader = _loader(train_dataset, args.batch_size, args.workers)
    validation_loader = _loader(validation_dataset, args.batch_size, args.workers)

    model = ReplayModel().to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    loss_function = ReplayLoss(stats["mean"], stats["std"]).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    start_epoch = 0
    best_validation_loss = float("inf")

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate
            group["initial_lr"] = args.learning_rate
        start_epoch = checkpoint["epoch"] + 1
        best_validation_loss = checkpoint.get("best_val_loss", float("inf"))

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    for _ in range(start_epoch):
        scheduler.step()

    print(f"Training on {device}")
    for epoch in range(start_epoch, args.epochs):
        started = time.perf_counter()
        teacher_forcing = max(0.0, 1.0 - epoch / 40.0)
        model.train()
        training_loss = 0.0

        progress = tqdm(train_loader, desc=f"Epoch {epoch} train", leave=False)
        for history, notes, environment, target in progress:
            history, notes, environment, target = [
                value.to(device, non_blocking=True)
                for value in (history, notes, environment, target)
            ]
            target = target[:, :TARGET_LENGTH, :]

            noise = torch.zeros_like(history)
            noise[..., POSITION_INDICES] = (
                torch.randn_like(history[..., POSITION_INDICES]) * 0.03
            )
            history = history + noise
            optimizer.zero_grad()
            with torch.amp.autocast(
                device_type=device.type, enabled=device.type == "cuda"
            ):
                prediction = model(
                    history,
                    notes,
                    environment,
                    target=target,
                    teacher_forcing=teacher_forcing,
                    steps=TARGET_LENGTH,
                )
                loss = loss_function(prediction, target[..., :27])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            training_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            progress = tqdm(
                validation_loader, desc=f"Epoch {epoch} validation", leave=False
            )
            for history, notes, environment, target in progress:
                history, notes, environment, target = [
                    value.to(device, non_blocking=True)
                    for value in (history, notes, environment, target)
                ]
                target = target[:, :TARGET_LENGTH, :]
                with torch.amp.autocast(
                    device_type=device.type, enabled=device.type == "cuda"
                ):
                    prediction = model(
                        history,
                        notes,
                        environment,
                        teacher_forcing=0.0,
                        steps=TARGET_LENGTH,
                    )
                    loss = loss_function(prediction, target[..., :27])
                validation_loss += loss.item()

        average_train = training_loss / len(train_loader)
        average_validation = validation_loss / len(validation_loader)
        elapsed = time.perf_counter() - started
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch}: train {average_train:.4f}, validation "
            f"{average_validation:.4f}, LR {current_lr:.2e}, "
            f"TF {teacher_forcing:.2f}, {elapsed:.1f}s"
        )
        scheduler.step()

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_validation_loss,
        }
        if average_validation < best_validation_loss:
            best_validation_loss = average_validation
            checkpoint["best_val_loss"] = best_validation_loss
            torch.save(checkpoint, output / "best.pth")
            torch.save(checkpoint, output / f"ep{epoch}.pth")
        elif epoch % 5 == 0:
            torch.save(checkpoint, output / f"ep{epoch}.pth")


def parse_args():
    parser = argparse.ArgumentParser(description="Train the replay motion model")
    parser.add_argument("--data", default="data/merged")
    parser.add_argument("--output", default="checkpoints/training")
    parser.add_argument("--resume")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--samples-per-epoch", type=int, default=DEFAULT_SAMPLES_PER_EPOCH
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
