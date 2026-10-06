"""Bidirectional gated recurrent unit over one whole clip.

A gated recurrent unit (GRU) is a small recurrent layer. It keeps a hidden
state and updates that state once per log-mel frame, in time order.

The context multilayer perceptron sees only 11 frames, about 220 milliseconds.
Events in this data often last longer than that, and a start time can be
wrong by more than the 200 millisecond collar even when the class is active
in the right region. This network reads all 497 frames of the 10 second clip,
so the hidden state can carry an event across seconds.

The layer is bidirectional. One GRU reads from the first frame toward the
last. A second GRU reads from the last frame toward the first. The two hidden
states are concatenated. The label of a frame can depend on what came before
it and on what comes after it. The two directions never cross into a
different clip: each forward call is one batch of separate sequences, and a
sequence is exactly one recording.

The class names stay the ten Domestic Environment Sound Event Detection
(DESED) names. Several classes may be active in the same frame. forward
returns logits, not probabilities. Binary cross-entropy with logits applies
the sigmoid inside the loss, which is more stable than a sigmoid followed by
binary cross-entropy on probabilities. There is no softmax, because a softmax
would force the ten classes to compete for one label.
"""

from __future__ import annotations

import torch
from torch import nn

from nndl_project.constants import N_MELS

# Hidden size in EACH direction. The linear layer sees both directions, so its
# input width is 2 * HIDDEN_UNITS. 64 is the planned comparison with the
# context MLP. A different size is a different experiment.
HIDDEN_UNITS = 64
LAYERS = 1
BIDIRECTIONAL = True


class FrameGRU(nn.Module):
    """One bidirectional GRU, then one logit per class at every frame.

    Input shape is (batch, frames, mel bands). For a full clip that is
    (batch, 497, 64). Output shape is (batch, frames, n_classes).

    The time axis is kept. Do not flatten the frames into independent rows.
    The MLP did that because each window was a separate example. Here the
    order of the frames is the input the GRU reads.
    """

    def __init__(self, n_mels: int = N_MELS, hidden: int = HIDDEN_UNITS, n_classes: int = 10) -> None:
        super().__init__()
        if hidden < 1 or n_mels < 1 or n_classes < 1:
            raise ValueError("hidden size, mel bands, and class count must be positive")
        self.n_mels = n_mels
        self.hidden = hidden
        self.n_classes = n_classes
        # batch_first matches the cache layout: clip, then frame, then band.
        # num_layers stays 1 so this run changes only the temporal memory, not
        # the depth. bias is on, which is the PyTorch default.
        self.gru = nn.GRU(
            input_size=n_mels,
            hidden_size=hidden,
            num_layers=LAYERS,
            batch_first=True,
            bidirectional=BIDIRECTIONAL,
        )
        # Both directions are kept, so the head input is 128 when hidden is 64.
        self.head = nn.Linear(hidden * 2, n_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError(f"expected (batch, frames, bands), got {tuple(features.shape)}")
        if features.shape[-1] != self.n_mels:
            raise ValueError(f"expected {self.n_mels} mel bands, got {features.shape[-1]}")
        # sequence has one vector per frame: (batch, frames, 2 * hidden).
        # The final hidden state is only the state after the last frame of each
        # direction. Scoring needs every frame, so the head reads `sequence`
        # and the final state is discarded.
        sequence, _final_state = self.gru(features)
        return self.head(sequence)
