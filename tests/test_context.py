"""Context windows stay inside one clip and the network has the planned shape."""

import numpy as np
import torch

from nndl_project.constants import N_FRAMES, N_MELS
from nndl_project.context import CONTEXT_RADIUS, HIDDEN_UNITS, ContextMLP, n_inputs, stack_context


def test_context_window_pads_the_edges_with_the_end_frame() -> None:
    frames = np.arange(N_FRAMES * N_MELS, dtype=np.float32).reshape(N_FRAMES, N_MELS)
    stacked = stack_context(frames, radius=CONTEXT_RADIUS)
    assert stacked.shape == (N_FRAMES, n_inputs(CONTEXT_RADIUS))
    assert stacked.dtype == np.float32
    first = stacked[0].reshape(2 * CONTEXT_RADIUS + 1, N_MELS)
    last = stacked[-1].reshape(2 * CONTEXT_RADIUS + 1, N_MELS)
    assert np.allclose(first[:CONTEXT_RADIUS], frames[0])
    assert np.allclose(first[CONTEXT_RADIUS], frames[0])
    assert np.allclose(first[CONTEXT_RADIUS + 1], frames[1])
    assert np.allclose(last[CONTEXT_RADIUS], frames[-1])
    assert np.allclose(last[CONTEXT_RADIUS + 1 :], frames[-1])
    middle = stacked[10].reshape(2 * CONTEXT_RADIUS + 1, N_MELS)
    assert np.allclose(middle[CONTEXT_RADIUS], frames[10])
    assert np.allclose(middle[0], frames[10 - CONTEXT_RADIUS])


def test_context_mlp_outputs_one_logit_per_class() -> None:
    model = ContextMLP(n_inputs(), HIDDEN_UNITS, n_classes=10)
    features = torch.zeros(4, n_inputs())
    logits = model(features)
    assert logits.shape == (4, 10)
    assert sum(parameter.numel() for parameter in model.parameters()) > 0
