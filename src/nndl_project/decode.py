"""Turn frame probabilities into timed events.

Boundaries sit on the 20 ms hop grid. The frame target uses the longer FFT
window, so a decoded edge can differ from the label by less than one window.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.ndimage import median_filter

from nndl_project.constants import CLASSES, HOP, MEDIAN_FRAMES, SAMPLE_RATE, THRESHOLD
from nndl_project.data import Event


def smooth_probabilities(probabilities: np.ndarray, median_frames: int = MEDIAN_FRAMES) -> np.ndarray:
    """Median-filter each class along time. Clips are filtered separately."""
    if probabilities.ndim != 2 or probabilities.shape[1] != len(CLASSES):
        raise ValueError(f"expected frame scores of shape (frames, {len(CLASSES)})")
    if median_frames < 1 or median_frames % 2 == 0:
        raise ValueError("median filter length must be a positive odd integer")
    if median_frames == 1:
        return np.asarray(probabilities, dtype=np.float32)
    smoothed = median_filter(np.asarray(probabilities, dtype=np.float32), size=(median_frames, 1), mode="nearest")
    return np.ascontiguousarray(smoothed, dtype=np.float32)


def decode_events(
    probabilities: np.ndarray,
    *,
    threshold: float | Sequence[float] = THRESHOLD,
    median_frames: int = MEDIAN_FRAMES,
) -> list[Event]:
    """Emit label, start, end, and score for each active run.

    The score is the mean smoothed probability inside the run. threshold is
    one number for every class, or one number per class in CLASSES order.
    """
    smoothed = smooth_probabilities(probabilities, median_frames)
    thresholds = class_thresholds(threshold)
    hop_seconds = HOP / SAMPLE_RATE
    events: list[Event] = []
    for class_index, label in enumerate(CLASSES):
        active = smoothed[:, class_index] >= thresholds[class_index]
        padded = np.pad(active.astype(np.int8), (1, 1))
        changes = np.diff(padded)
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)
        for start, end in zip(starts.tolist(), ends.tolist(), strict=True):
            score = float(smoothed[start:end, class_index].mean())
            events.append(
                Event(
                    label=label,
                    onset=start * hop_seconds,
                    offset=end * hop_seconds,
                    score=score,
                )
            )
    return events


def class_thresholds(threshold: float | Sequence[float]) -> np.ndarray:
    """Return one threshold per class. A single number is repeated."""
    values = np.asarray(threshold, dtype=np.float64)
    if values.ndim == 0:
        values = np.full(len(CLASSES), float(values))
    if values.shape != (len(CLASSES),):
        raise ValueError(f"expected one threshold or {len(CLASSES)}, got shape {values.shape}")
    if not np.all((values > 0.0) & (values < 1.0)):
        raise ValueError("every threshold must be strictly between 0 and 1")
    return values
