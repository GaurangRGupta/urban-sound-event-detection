"""The street label map is complete, fixed, and merges overlaps correctly."""

import pytest

from nndl_project.constants import CLASSES
from nndl_project.data import Event
from nndl_project.label_map import (
    DESED_TO_STREET,
    STREET_CLASSES,
    STREET_CLASSES_WITH_TIMESTAMPS,
    STREET_CLASSES_WITHOUT_TIMESTAMPS,
    check_map,
    to_street_events,
    to_street_label,
)


def test_the_ten_street_classes_are_the_problem_statement_list() -> None:
    assert STREET_CLASSES == (
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


def test_every_desed_label_lands_on_one_street_class() -> None:
    check_map()
    assert set(DESED_TO_STREET) == set(CLASSES)
    for name in CLASSES:
        assert to_street_label(name) in STREET_CLASSES


def test_the_map_is_the_one_written_in_the_problem_statement() -> None:
    assert to_street_label("Speech") == "speech_crowd"
    assert to_street_label("Dog") == "dog"
    others = [n for n in CLASSES if n not in {"Speech", "Dog"}]
    assert len(others) == 8
    assert all(to_street_label(n) == "other" for n in others)


def test_classes_without_timestamps_are_the_seven_street_only_names() -> None:
    assert STREET_CLASSES_WITH_TIMESTAMPS == ("speech_crowd", "dog", "other")
    assert set(STREET_CLASSES_WITHOUT_TIMESTAMPS) == {
        "car_horn",
        "siren",
        "engine_traffic",
        "two_wheeler",
        "construction",
        "music",
        "bird",
    }


def test_unknown_label_is_an_error() -> None:
    with pytest.raises(ValueError):
        to_street_label("Car_horn")


def test_overlapping_other_events_merge_and_keep_the_best_score() -> None:
    events = [
        Event("Dishes", 1.0, 3.0, 0.6),
        Event("Running_water", 2.5, 5.0, 0.9),
        Event("Frying", 7.0, 8.0, 0.7),
        Event("Speech", 0.0, 2.0, 0.8),
    ]
    street = to_street_events(events)
    assert Event("speech_crowd", 0.0, 2.0, 0.8) in street
    others = sorted((e for e in street if e.label == "other"), key=lambda e: e.onset)
    assert others == [Event("other", 1.0, 5.0, 0.9), Event("other", 7.0, 8.0, 0.7)]


def test_reference_events_keep_no_score() -> None:
    street = to_street_events([Event("Blender", 0.0, 1.0), Event("Cat", 0.5, 2.0)])
    assert street == [Event("other", 0.0, 2.0, None)]
