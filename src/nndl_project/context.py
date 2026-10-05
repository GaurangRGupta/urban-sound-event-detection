"""Context window and the shared multilayer perceptron.

Each example is the log-mel frame being classified, plus the frames on either
side. The clip edges repeat the first or last frame so every frame still has
a full window. The window is built per clip, so context never crosses from
one recording into another.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from nndl_project.constants import N_MELS

CONTEXT_RADIUS = 5
HIDDEN_UNITS = 128


def context_frames(radius: int = CONTEXT_RADIUS) -> int:
    if radius < 0:
        raise ValueError("context radius must be zero or more")
    return 2 * radius + 1


def n_inputs(radius: int = CONTEXT_RADIUS, n_mels: int = N_MELS) -> int:
    return context_frames(radius) * n_mels


def stack_context(frames: np.ndarray, radius: int = CONTEXT_RADIUS) -> np.ndarray:
    """Flatten a neighborhood around every frame.

    frames has shape (n_frames, n_mels). The result has shape
    (n_frames, (2 * radius + 1) * n_mels), with the center frame in the middle
    of each neighborhood.
    """
    if radius < 0:
        raise ValueError("context radius must be zero or more")
    if frames.ndim != 2:
        raise ValueError(f"expected one clip of shape (frames, bands), got {frames.shape}")
    if radius == 0:
        return np.ascontiguousarray(frames, dtype=np.float32)
    padded = np.pad(frames, ((radius, radius), (0, 0)), mode="edge")
    view = np.lib.stride_tricks.sliding_window_view(padded, context_frames(radius), axis=0)
    # view is (n_frames, n_mels, context). Put time before bands, then flatten.
    ordered = np.moveaxis(view, -1, 1)
    n_frames = frames.shape[0]
    flat = ordered.reshape(n_frames, -1)
    return np.ascontiguousarray(flat, dtype=np.float32)


class ContextMLP(nn.Module):
    """One dense hidden layer and one sigmoid probability per class.

    forward returns logits. The training loss applies the sigmoid itself.
    """

    def __init__(self, n_inputs: int, hidden: int = HIDDEN_UNITS, n_classes: int = 10) -> None:
        super().__init__()
        if hidden < 1:
            raise ValueError("hidden size must be positive")
        self.n_inputs = n_inputs
        self.hidden = hidden
        self.n_classes = n_classes
        self.net = nn.Sequential(
            nn.Linear(n_inputs, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)
