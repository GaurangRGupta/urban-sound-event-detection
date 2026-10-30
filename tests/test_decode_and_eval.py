"""Per-class thresholds, street-level metrics, and the lenient held-out reader."""

from pathlib import Path

import numpy as np
import pytest

from nndl_project.constants import CLASSES, N_FRAMES
from nndl_project.data import Event
from nndl_project.decode import class_thresholds, decode_events
from nndl_project.label_map import STREET_CLASSES
from nndl_project.metrics import event_counts_for_clip, segment_counts_for_clip


def test_one_threshold_or_one_per_class() -> None:
    assert class_thresholds(0.5).shape == (len(CLASSES),)
    assert np.allclose(class_thresholds([0.3] * len(CLASSES)), 0.3)
    with pytest.raises(ValueError):
        class_thresholds([0.5, 0.5])
    with pytest.raises(ValueError):
        class_thresholds(1.0)


def test_per_class_threshold_changes_only_that_class() -> None:
    probs = np.zeros((N_FRAMES, len(CLASSES)), dtype=np.float32)
    probs[100:200, :] = 0.4
    at_half = decode_events(probs, threshold=0.5)
    assert at_half == []
    thresholds = [0.5] * len(CLASSES)
    speech = CLASSES.index("Speech")
    thresholds[speech] = 0.3
    events = decode_events(probs, threshold=thresholds)
    assert [e.label for e in events] == ["Speech"]
    assert events[0].onset == pytest.approx(100 * 0.02)
    assert events[0].offset == pytest.approx(200 * 0.02)


def test_metrics_score_street_names() -> None:
    reference = [Event("speech_crowd", 1.0, 3.0), Event("other", 4.0, 6.0)]
    prediction = [Event("speech_crowd", 1.1, 3.1, 0.9), Event("dog", 4.0, 6.0, 0.8)]
    events = event_counts_for_clip(reference, prediction, classes=STREET_CLASSES)
    assert (events["speech_crowd"].tp, events["speech_crowd"].fp, events["speech_crowd"].fn) == (1, 0, 0)
    assert events["other"].fn == 1
    assert events["dog"].fp == 1
    segments = segment_counts_for_clip(reference, prediction, classes=STREET_CLASSES)
    assert segments["speech_crowd"].tp >= 2


def test_lenient_reader_maps_jams_and_drops_bad_rows(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from nndl_project.evaluate import read_labels_lenient

    tsv = tmp_path / "labels.tsv"
    tsv.write_text(
        "filename\tonset\toffset\tevent_label\n"
        "1.jams\t0.5\t2.0\tDog\n"
        "1.jams\t3.0\t12.0\tSpeech\n"
        "2.jams\t\t\t\n"
        "3.wav\t2.0\t1.0\tCat\n"
        "3.wav\t1.0\t2.0\tCar_horn\n",
        encoding="utf-8",
    )
    labels, dropped = read_labels_lenient(tsv)
    assert labels["1.wav"] == [Event("Dog", 0.5, 2.0), Event("Speech", 3.0, 10.0)]
    assert labels["2.wav"] == []
    assert labels["3.wav"] == []
    assert dropped == 2


def test_tuning_finds_a_planted_threshold() -> None:
    pytest.importorskip("torch")
    from nndl_project.constants import CLASSES as NAMES
    from nndl_project.data import frame_targets
    from nndl_project.evaluate import tune_decode

    rng = np.random.default_rng(0)
    references, probabilities = [], []
    speech = NAMES.index("Speech")
    for _ in range(30):
        onset = float(rng.uniform(0, 6))
        events = [Event("Speech", onset, onset + float(rng.uniform(1, 3)))]
        targets = frame_targets(events).astype(np.float32)
        probs = np.full(targets.shape, 0.02, dtype=np.float32)
        # Speech frames sit at 0.35 and silence at 0.25, so only 0.3 separates them.
        probs[:, speech] = np.where(targets[:, speech] > 0, 0.35, 0.25)
        probabilities.append(probs)
        references.append(events)
    best = tune_decode(np.stack(probabilities), references)
    assert best["thresholds"]["Speech"] == 0.3
    # Classes that never occur keep the default rather than the lowest grid value.
    assert best["thresholds"]["Cat"] == 0.5
