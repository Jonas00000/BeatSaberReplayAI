import random

import torch
from torch import nn


class ReplayModel(nn.Module):
    """Encode motion history and map context, then decode one motion chunk."""

    def __init__(
        self,
        input_dim=28,
        output_dim=27,
        note_dim=21,
        environment_dim=5,
        hidden_size=768,
    ):
        super().__init__()
        self.environment_dim = environment_dim
        self.note_size = 128
        self.environment_size = 32
        self.target_fps = 60.0

        self.motion_encoder = nn.GRU(
            input_dim, hidden_size, num_layers=2, batch_first=True, dropout=0.3
        )
        self.note_embedding = nn.Sequential(
            nn.Linear(note_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.note_size),
        )
        self.note_encoder = nn.GRU(
            self.note_size,
            self.note_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.wall_embedding = nn.Sequential(
            nn.Linear(environment_dim, 16),
            nn.ReLU(),
            nn.Linear(16, self.environment_size),
            nn.ReLU(),
        )
        self.attn = nn.MultiheadAttention(
            self.note_size * 2, num_heads=4, batch_first=True, dropout=0.3
        )

        decoder_input_dim = input_dim + self.note_size * 2 + 1 + self.environment_size
        self.norm = nn.LayerNorm(decoder_input_dim)
        self.decoder_gru = nn.GRU(
            decoder_input_dim,
            hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
        )
        self.query_proj = nn.Linear(
            hidden_size + 1 + self.environment_size, self.note_size * 2
        )
        self.output_layer = nn.Linear(hidden_size, output_dim)

    def forward(
        self,
        history,
        notes,
        environment,
        target=None,
        teacher_forcing=0.5,
        steps=30,
    ):
        batch_size = history.size(0)
        _, hidden = self.motion_encoder(history)

        padding_mask = notes[:, :, 1:5].sum(dim=2) == 0
        # Fully masked attention produces NaNs; empty contexts keep their padded encodings.
        all_padded = padding_mask.all(dim=1, keepdim=True)
        padding_mask = padding_mask & ~all_padded
        note_embeddings = self.note_embedding(notes)
        note_context, _ = self.note_encoder(note_embeddings)

        # Each chunk starts from the encoded motion history, not a carried decoder state.
        current_frame = history[:, -1, :]
        height = history[:, 0, 27:28]
        outputs = []

        for step in range(steps):
            current_time = step / self.target_fps
            time_feature = torch.full(
                (batch_size, 1), current_time, device=history.device
            )
            if step < environment.size(1):
                frame_environment = environment[:, step, :]
            else:
                frame_environment = torch.zeros(
                    (batch_size, self.environment_dim), device=history.device
                )
            environment_embedding = self.wall_embedding(frame_environment)

            query_input = torch.cat(
                [hidden[-1], time_feature, environment_embedding], dim=1
            )
            query = self.query_proj(query_input).unsqueeze(1)
            context, _ = self.attn(
                query,
                note_context,
                note_context,
                key_padding_mask=padding_mask,
            )
            context = context.squeeze(1)

            decoder_input = torch.cat(
                [current_frame, context, time_feature, environment_embedding], dim=1
            )
            decoder_input = self.norm(decoder_input).unsqueeze(1)
            decoded, hidden = self.decoder_gru(decoder_input, hidden)
            prediction = self.output_layer(decoded.squeeze(1))
            outputs.append(prediction)

            prediction_with_height = torch.cat([prediction, height], dim=1)
            if (
                target is not None
                and random.random() < teacher_forcing
                and step < target.size(1)
            ):
                current_frame = target[:, step, :]
            else:
                current_frame = prediction_with_height

        return torch.stack(outputs, dim=1)
