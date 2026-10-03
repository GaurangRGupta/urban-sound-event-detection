"""Segment F1 at 1 second bins, and event F1 with a 200 ms collar.

Event matching is one to one inside a clip and a class. A prediction matches
a reference when the onset is within 200 ms and the offset is within
max(200 ms, 20 percent of the reference duration). Counts are summed over clips.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from nndl_project.constants import (
    CLASSES,
    CLIP_SECONDS,
    OFFSET_COLLAR_FRACTION,
    OFFSET_COLLAR_SECONDS,
    ONSET_COLLAR_SECONDS,
    SEGMENT_SECONDS,
)
from nndl_project.data import Event


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, other: Counts) -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn


def empty_counts() -> dict[str, Counts]:
    return {name: Counts() for name in CLASSES}


def precision_recall_f1(counts: Counts) -> tuple[float, float, float] | None:
    """Return precision, recall, and F1. None when there is nothing to score."""
    if counts.tp + counts.fp + counts.fn == 0:
        return None
    precision = counts.tp / (counts.tp + counts.fp) if (counts.tp + counts.fp) else 0.0
    recall = counts.tp / (counts.tp + counts.fn) if (counts.tp + counts.fn) else 0.0
    if precision + recall == 0:
        return precision, recall, 0.0
    return precision, recall, 2.0 * precision * recall / (precision + recall)


def micro_counts(by_class: dict[str, Counts]) -> Counts:
    total = Counts()
    for counts in by_class.values():
        total.add(counts)
    return total


def macro_f1(by_class: dict[str, Counts]) -> float | None:
    scores = []
    for name in CLASSES:
        row = precision_recall_f1(by_class[name])
        if row is not None:
            scores.append(row[2])
    if not scores:
        return None
    return float(np.mean(scores))


def _by_class(events: list[Event]) -> dict[str, list[Event]]:
    grouped = {name: [] for name in CLASSES}
    for event in events:
        grouped[event.label].append(event)
    return grouped


def segment_counts_for_clip(
    reference: list[Event],
    prediction: list[Event],
    *,
    clip_seconds: float = CLIP_SECONDS,
    segment_seconds: float = SEGMENT_SECONDS,
) -> dict[str, Counts]:
    """A 1 second bin is positive when any event of that class overlaps it."""
    if clip_seconds <= 0 or segment_seconds <= 0:
        raise ValueError("segment length must be positive")
    n_segments = int(np.ceil(clip_seconds / segment_seconds))
    edges = np.arange(n_segments + 1, dtype=np.float64) * segment_seconds
    counts = empty_counts()
    reference_groups = _by_class(reference)
    prediction_groups = _by_class(prediction)
    for label in CLASSES:
        ref_active = _segment_activity(reference_groups[label], edges)
        pred_active = _segment_activity(prediction_groups[label], edges)
        counts[label].tp = int(np.count_nonzero(ref_active & pred_active))
        counts[label].fp = int(np.count_nonzero(~ref_active & pred_active))
        counts[label].fn = int(np.count_nonzero(ref_active & ~pred_active))
    return counts


def _segment_activity(events: list[Event], edges: np.ndarray) -> np.ndarray:
    active = np.zeros(len(edges) - 1, dtype=bool)
    for event in events:
        hit = (event.onset < edges[1:]) & (event.offset > edges[:-1])
        active |= hit
    return active


def event_counts_for_clip(
    reference: list[Event],
    prediction: list[Event],
    *,
    onset_collar: float = ONSET_COLLAR_SECONDS,
    offset_collar: float = OFFSET_COLLAR_SECONDS,
    offset_fraction: float = OFFSET_COLLAR_FRACTION,
) -> dict[str, Counts]:
    counts = empty_counts()
    reference_groups = _by_class(reference)
    prediction_groups = _by_class(prediction)
    for label in CLASSES:
        counts[label] = _match_one_class(
            reference_groups[label],
            prediction_groups[label],
            onset_collar=onset_collar,
            offset_collar=offset_collar,
            offset_fraction=offset_fraction,
        )
    return counts


def _match_one_class(
    reference: list[Event],
    prediction: list[Event],
    *,
    onset_collar: float,
    offset_collar: float,
    offset_fraction: float,
) -> Counts:
    n_ref = len(reference)
    n_pred = len(prediction)
    if n_ref == 0 and n_pred == 0:
        return Counts()
    if n_ref == 0:
        return Counts(fp=n_pred)
    if n_pred == 0:
        return Counts(fn=n_ref)
    cost = np.ones((n_ref, n_pred), dtype=np.float64)
    for ref_index, ref in enumerate(reference):
        duration = ref.offset - ref.onset
        allowed_offset = max(offset_collar, offset_fraction * duration)
        for pred_index, pred in enumerate(prediction):
            onset_ok = abs(pred.onset - ref.onset) <= onset_collar
            offset_ok = abs(pred.offset - ref.offset) <= allowed_offset
            if onset_ok and offset_ok:
                cost[ref_index, pred_index] = 0.0
    rows, cols = linear_sum_assignment(cost)
    true_positives = int(np.count_nonzero(cost[rows, cols] == 0.0))
    return Counts(tp=true_positives, fp=n_pred - true_positives, fn=n_ref - true_positives)


def accumulate(total: dict[str, Counts], clip_counts: dict[str, Counts]) -> None:
    for label in CLASSES:
        total[label].add(clip_counts[label])
