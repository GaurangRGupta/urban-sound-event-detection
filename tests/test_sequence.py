"""The frame GRU scores every frame and can see both directions."""

import torch

from nndl_project.constants import N_FRAMES, N_MELS
from nndl_project.sequence import HIDDEN_UNITS, FrameGRU


def test_frame_gru_outputs_one_logit_per_class_at_every_frame() -> None:
    model = FrameGRU(n_mels=N_MELS, hidden=HIDDEN_UNITS, n_classes=10)
    features = torch.zeros(2, N_FRAMES, N_MELS)
    logits = model(features)
    assert logits.shape == (2, N_FRAMES, 10)
    assert torch.isfinite(logits).all()
    # Bidirectional: the recurrent output width is the two hidden states.
    assert model.gru.bidirectional is True
    assert model.gru.num_layers == 1
    assert model.gru.hidden_size == HIDDEN_UNITS
    assert model.head.in_features == HIDDEN_UNITS * 2
    # One direction: 3 gates * (input->hidden + hidden->hidden + two biases).
    # Both directions, plus the linear head from 128 to 10.
    one_direction = 3 * HIDDEN_UNITS * (N_MELS + HIDDEN_UNITS + 2)
    expected = 2 * one_direction + (HIDDEN_UNITS * 2 * 10 + 10)
    assert sum(parameter.numel() for parameter in model.parameters()) == expected == 51210


def test_a_later_frame_can_change_an_earlier_logit() -> None:
    # The backward GRU is the path from a later frame back to an earlier one.
    # Two frames are enough to see that path. A 497-frame silence can hide it,
    # because a fresh network does not yet keep a signal across the whole clip.
    torch.manual_seed(0)
    model = FrameGRU(n_mels=N_MELS, hidden=HIDDEN_UNITS, n_classes=10)
    model.eval()
    assert model.gru.weight_ih_l0_reverse.shape[1] == N_MELS
    base = torch.zeros(1, 2, N_MELS)
    later = base.clone()
    later[0, 1, :] = 5.0
    with torch.no_grad():
        early = model(base)[0, 0]
        early_after_change = model(later)[0, 0]
    assert not torch.allclose(early, early_after_change)


def test_one_clip_does_not_change_another_clips_logits() -> None:
    torch.manual_seed(0)
    model = FrameGRU(n_mels=N_MELS, hidden=HIDDEN_UNITS, n_classes=10)
    model.eval()
    batch = torch.randn(2, N_FRAMES, N_MELS)
    swapped = batch.clone()
    swapped[1] = torch.randn(N_FRAMES, N_MELS)
    with torch.no_grad():
        first = model(batch)[0]
        first_again = model(swapped)[0]
    assert torch.allclose(first, first_again)
