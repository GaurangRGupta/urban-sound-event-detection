"""Small convolutional recurrent network over one whole clip.

A convolutional recurrent neural network (CRNN) is the model that was
written down at the start of the project and then set aside. The
convolutional front end looks at the log-mel matrix as a grid: time in one
direction, frequency band in the other. A gated recurrent unit (GRU) then
reads that grid in time order and scores every frame.

This is not a pretrained photo network. The filters are learned from the
synthetic clips. There is no dropout and no batch normalization.

Frequency is pooled. Time is not. The clip still has 497 frames, so the same
median filter and the same threshold of 0.5 can turn frame scores into
events. Context does not cross from one recording into another. Each forward
call is one batch of separate clips.

forward returns logits, not probabilities. Binary cross-entropy with logits
applies the sigmoid inside the loss. There is no softmax, because more than
one of the ten classes may be active in the same frame.
"""

from __future__ import annotations

import torch
from torch import nn

from nndl_project.constants import N_FRAMES, N_MELS

# First convolution writes this many channels. The second writes twice that.
# 16 then 32 is the planned comparison. Another width is a different run.
CONV_CHANNELS = 16
# Pool frequency by 4, twice. 64 bands become 16, then 4. The kernel is
# (1, 4): 1 leaves the time axis alone, 4 shrinks the frequency axis.
FREQ_POOL = 4
# Hidden size in EACH direction of the GRU. The linear layer sees both
# directions, so its input width is 2 * HIDDEN_UNITS.
HIDDEN_UNITS = 64
LAYERS = 1
BIDIRECTIONAL = True


class SmallCRNN(nn.Module):
    """Two convolutions, frequency pooling, one bidirectional GRU, then logits.

    Input shape is (batch, frames, mel bands). For a full clip that is
    (batch, 497, 64). Output shape is (batch, frames, n_classes).

    Inside the module the grid is (batch, 1 channel, frames, bands). That
    extra channel is how a convolution reads a single log-mel matrix. It is
    not a picture file.
    """

    def __init__(
        self,
        n_mels: int = N_MELS,
        n_frames: int = N_FRAMES,
        channels: int = CONV_CHANNELS,
        hidden: int = HIDDEN_UNITS,
        n_classes: int = 10,
    ) -> None:
        super().__init__()
        if channels < 1 or hidden < 1 or n_mels < 1 or n_classes < 1:
            raise ValueError("channels, hidden size, mel bands, and class count must be positive")
        # Two pools of 4 must land on a whole number of bands.
        if n_mels % (FREQ_POOL * FREQ_POOL) != 0:
            raise ValueError(f"{n_mels} mel bands cannot be pooled by {FREQ_POOL} twice")
        self.n_mels = n_mels
        self.n_frames = n_frames
        self.channels = channels
        self.hidden = hidden
        self.n_classes = n_classes
        # padding 1 with a 3 by 3 kernel keeps the time and frequency sizes
        # unchanged. The pool, not the convolution, shrinks frequency.
        self.conv1 = nn.Conv2d(1, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(1, FREQ_POOL))
        self.relu = nn.ReLU()
        bands_after_pool = n_mels // (FREQ_POOL * FREQ_POOL)
        gru_input = (channels * 2) * bands_after_pool
        self.gru_input = gru_input
        self.gru = nn.GRU(
            input_size=gru_input,
            hidden_size=hidden,
            num_layers=LAYERS,
            batch_first=True,
            bidirectional=BIDIRECTIONAL,
        )
        self.head = nn.Linear(hidden * 2, n_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError(f"expected (batch, frames, bands), got {tuple(features.shape)}")
        if features.shape[-1] != self.n_mels:
            raise ValueError(f"expected {self.n_mels} mel bands, got {features.shape[-1]}")
        # (batch, frames, bands) -> (batch, 1, frames, bands).
        grid = features.unsqueeze(1)
        grid = self.pool(self.relu(self.conv1(grid)))
        grid = self.pool(self.relu(self.conv2(grid)))
        if grid.shape[2] != features.shape[1]:
            raise RuntimeError(
                f"time axis changed from {features.shape[1]} frames to {grid.shape[2]}. "
                "Frequency pooling must not shorten the clip."
            )
        # (batch, channels, frames, bands) -> (batch, frames, channels * bands).
        batch, conv_channels, frames, bands = grid.shape
        sequence = grid.permute(0, 2, 1, 3).contiguous().reshape(batch, frames, conv_channels * bands)
        if sequence.shape[-1] != self.gru_input:
            raise RuntimeError(f"GRU expected {self.gru_input} inputs, got {sequence.shape[-1]}")
        # sequence is one vector per frame. The final hidden state is only
        # the state after the last frame. Scoring needs every frame, so the
        # head reads `hidden` and the final state is discarded.
        hidden, _final_state = self.gru(sequence)
        return self.head(hidden)
