"""Hand-checked frame labels, decode, and the two F1 definitions."""

import numpy as np
import pytest

from nndl_project.constants import CLASSES, HOP, N_FFT, N_FRAMES, N_MELS, SAMPLE_RATE, TARGET_SAMPLES
from nndl_project.data import Event, frame_targets
from nndl_project.decode import decode_events
from nndl_project.features import log_mel, mel_filterbank
from nndl_project.metrics import event_counts_for_clip, precision_recall_f1, segment_counts_for_clip


def test_frame_is_positive_only_when_the_window_overlaps_the_label() -> None:
    event = Event("Speech", 1.0, 1.2)
    targets = frame_targets([event])
    starts = np.arange(N_FRAMES) * (HOP / SAMPLE_RATE)
    ends = starts + (N_FFT / SAMPLE_RATE)
    expected = (event.onset < ends) & (event.offset > starts)
    speech = CLASSES.index("Speech")
    assert int(expected.sum()) == 13
    assert np.array_equal(targets[:, speech].astype(bool), expected)
    assert int(targets[:, CLASSES.index("Dog")].sum()) == 0
    covered = frame_targets([Event("Dog", 0.0, 10.0)])
    assert int(covered[:, CLASSES.index("Dog")].sum()) == N_FRAMES


def test_unknown_label_is_rejected() -> None:
    with pytest.raises(ValueError):
        frame_targets([Event("car_horn", 0.0, 1.0)])


def test_log_mel_shape_on_a_sine() -> None:
    time = np.arange(TARGET_SAMPLES, dtype=np.float32) / SAMPLE_RATE
    waveform = (0.1 * np.sin(2.0 * np.pi * 440.0 * time)).astype(np.float32)
    features = log_mel(waveform)
    assert features.shape == (N_FRAMES, N_MELS)
    assert features.dtype == np.float32
    assert np.isfinite(features).all()
    bank = mel_filterbank()
    assert bank.shape == (N_MELS, N_FFT // 2 + 1)
    assert np.all(bank.sum(axis=1) > 0)


def test_segment_f1_on_a_two_second_clip() -> None:
    reference = [Event("Speech", 0.0, 1.0)]
    prediction = [Event("Speech", 0.0, 2.0, score=0.7)]
    counts = segment_counts_for_clip(reference, prediction, clip_seconds=2.0, segment_seconds=1.0)
    speech = counts["Speech"]
    assert (speech.tp, speech.fp, speech.fn) == (1, 1, 0)
    row = precision_recall_f1(speech)
    assert row is not None
    precision, recall, f1 = row
    assert precision == pytest.approx(0.5)
    assert recall == pytest.approx(1.0)
    assert f1 == pytest.approx(2.0 / 3.0)
    assert precision_recall_f1(counts["Dog"]) is None


def test_event_collar_is_one_to_one() -> None:
    reference = [Event("Speech", 1.0, 2.0)]
    matched = [Event("Speech", 1.1, 2.05, score=0.8)]
    hit = event_counts_for_clip(reference, matched)["Speech"]
    assert (hit.tp, hit.fp, hit.fn) == (1, 0, 0)

    long_reference = [Event("Speech", 0.0, 2.0)]
    inside_offset = event_counts_for_clip(long_reference, [Event("Speech", 0.0, 2.3)])["Speech"]
    outside_offset = event_counts_for_clip(long_reference, [Event("Speech", 0.0, 2.5)])["Speech"]
    assert inside_offset.tp == 1
    assert (outside_offset.tp, outside_offset.fp, outside_offset.fn) == (0, 1, 1)

    two_predictions = [
        Event("Speech", 0.05, 2.05),
        Event("Speech", 0.0, 2.1),
    ]
    one_match = event_counts_for_clip(long_reference, two_predictions)["Speech"]
    assert (one_match.tp, one_match.fp, one_match.fn) == (1, 1, 0)


def test_decode_emits_a_timed_event_and_drops_a_one_frame_spike() -> None:
    probabilities = np.zeros((N_FRAMES, len(CLASSES)), dtype=np.float32)
    probabilities[10:20, 0] = 0.9
    events = decode_events(probabilities, threshold=0.5, median_frames=1)
    assert len(events) == 1
    assert events[0].label == CLASSES[0]
    assert events[0].onset == pytest.approx(10 * HOP / SAMPLE_RATE)
    assert events[0].offset == pytest.approx(20 * HOP / SAMPLE_RATE)
    assert events[0].score == pytest.approx(0.9)

    spike = np.zeros_like(probabilities)
    spike[10, 0] = 0.9
    assert decode_events(spike, threshold=0.5, median_frames=11) == []
