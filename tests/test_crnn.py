"""The small CRNN keeps every frame and does not mix clips."""

import torch

from nndl_project.constants import N_FRAMES, N_MELS
from nndl_project.crnn import CONV_CHANNELS, FREQ_POOL, HIDDEN_UNITS, SmallCRNN


def test_crnn_outputs_one_logit_per_class_at_every_frame() -> None:
    model = SmallCRNN()
    features = torch.zeros(2, N_FRAMES, N_MELS)
    logits = model(features)
    assert logits.shape == (2, N_FRAMES, 10)
    assert torch.isfinite(logits).all()
    # 3 by 3 filters, then a bidirectional GRU of hidden size 64.
    assert tuple(model.conv1.weight.shape) == (CONV_CHANNELS, 1, 3, 3)
    assert tuple(model.conv2.weight.shape) == (CONV_CHANNELS * 2, CONV_CHANNELS, 3, 3)
    assert tuple(model.pool.kernel_size) == (1, FREQ_POOL)
    assert model.gru.bidirectional is True
    assert model.gru.num_layers == 1
    assert model.gru.hidden_size == HIDDEN_UNITS
    # 64 bands pooled by 4 twice is 4 bands. 32 channels times 4 bands is 128.
    assert model.gru.input_size == 128
    assert model.head.in_features == HIDDEN_UNITS * 2
    conv1 = CONV_CHANNELS * 1 * 3 * 3 + CONV_CHANNELS
    conv2 = (CONV_CHANNELS * 2) * CONV_CHANNELS * 3 * 3 + (CONV_CHANNELS * 2)
    one_direction = 3 * HIDDEN_UNITS * (128 + HIDDEN_UNITS + 2)
    head = HIDDEN_UNITS * 2 * 10 + 10
    expected = conv1 + conv2 + 2 * one_direction + head
    assert sum(parameter.numel() for parameter in model.parameters()) == expected == 80586


def test_one_clip_does_not_change_another_clips_logits() -> None:
    torch.manual_seed(0)
    model = SmallCRNN()
    model.eval()
    batch = torch.randn(2, N_FRAMES, N_MELS)
    swapped = batch.clone()
    swapped[1] = torch.randn(N_FRAMES, N_MELS)
    with torch.no_grad():
        first = model(batch)[0]
        first_again = model(swapped)[0]
    assert torch.allclose(first, first_again)
