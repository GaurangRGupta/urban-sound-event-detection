"""The fixed street class list and the map from the DESED labels onto it.

The problem statement fixes ten output classes. DESED, the only dataset here
with timestamps, has ten home classes. The map is written down once, here,
and a test checks that every DESED label lands on exactly one street class.

Speech maps to speech_crowd. Dog maps to dog. The other eight DESED names map
to other. A blender is not traffic, so nothing is forced onto a street class
it does not belong to. The seven street classes that receive no DESED label
have no timestamp data in this project, and they are reported as such.
"""

from __future__ import annotations

from nndl_project.constants import CLASSES as DESED_CLASSES
from nndl_project.data import Event

STREET_CLASSES: tuple[str, ...] = (
    "speech_crowd",
    "car_horn",
    "siren",
    "engine_traffic",
    "two_wheeler",
    "construction",
    "dog",
    "music",
    "bird",
    "other",
)

DESED_TO_STREET: dict[str, str] = {
    "Alarm_bell_ringing": "other",
    "Blender": "other",
    "Cat": "other",
    "Dishes": "other",
    "Dog": "dog",
    "Electric_shaver_toothbrush": "other",
    "Frying": "other",
    "Running_water": "other",
    "Speech": "speech_crowd",
    "Vacuum_cleaner": "other",
}

# Street classes that at least one DESED label lands on. Only these have
# strong (timestamped) labels, so only these get an event-F1 score.
STREET_CLASSES_WITH_TIMESTAMPS: tuple[str, ...] = tuple(
    name for name in STREET_CLASSES if name in set(DESED_TO_STREET.values())
)
STREET_CLASSES_WITHOUT_TIMESTAMPS: tuple[str, ...] = tuple(
    name for name in STREET_CLASSES if name not in STREET_CLASSES_WITH_TIMESTAMPS
)


def to_street_label(desed_label: str) -> str:
    """Return the street class for one DESED label. Unknown names are an error."""
    try:
        return DESED_TO_STREET[desed_label]
    except KeyError as exc:
        raise ValueError(f"{desed_label!r} is not a DESED class") from exc


def to_street_events(events: list[Event]) -> list[Event]:
    """Relabel DESED events as street events and merge overlaps of one class.

    Several DESED classes land on `other`. Two of them can overlap in time,
    for example Dishes and Running_water. After relabeling they are the same
    class, so overlapping or touching intervals are merged into one event.
    The merged score is the highest score of the pieces. Reference events
    have no score, and the merged reference keeps None.
    """
    by_label: dict[str, list[Event]] = {}
    for event in events:
        by_label.setdefault(to_street_label(event.label), []).append(event)
    merged: list[Event] = []
    for label in STREET_CLASSES:
        pieces = sorted(by_label.get(label, []), key=lambda e: (e.onset, e.offset))
        current: Event | None = None
        for piece in pieces:
            if current is None:
                current = Event(label, piece.onset, piece.offset, piece.score)
                continue
            if piece.onset <= current.offset:
                score = _max_score(current.score, piece.score)
                current = Event(label, current.onset, max(current.offset, piece.offset), score)
            else:
                merged.append(current)
                current = Event(label, piece.onset, piece.offset, piece.score)
        if current is not None:
            merged.append(current)
    return merged


def _max_score(a: float | None, b: float | None) -> float | None:
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b)


def check_map() -> None:
    """Raise if the map is not a complete function from DESED onto street names."""
    if set(DESED_TO_STREET) != set(DESED_CLASSES):
        missing = set(DESED_CLASSES) - set(DESED_TO_STREET)
        extra = set(DESED_TO_STREET) - set(DESED_CLASSES)
        raise ValueError(f"label map is incomplete: missing {sorted(missing)}, extra {sorted(extra)}")
    bad = {name: street for name, street in DESED_TO_STREET.items() if street not in STREET_CLASSES}
    if bad:
        raise ValueError(f"label map points outside the street classes: {bad}")
    if len(STREET_CLASSES) != 10 or len(set(STREET_CLASSES)) != 10:
        raise ValueError("the street class list must be ten distinct names")
