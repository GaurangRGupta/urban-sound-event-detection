"""Strong-label reads for the synthetic DESED splits.

Train and validation both contain a file named 0.wav. Those are different
clips. A label is applied only to the wav that lives in the same split folder.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nndl_project.constants import CLASSES, HOP, N_FFT, N_FRAMES, SAMPLE_RATE


@dataclass(frozen=True)
class Event:
    label: str
    onset: float
    offset: float
    score: float | None = None


def project_root() -> Path:
    """Return the repository root that holds pyproject.toml and this package."""
    here = Path(__file__).resolve()
    starts = [Path.cwd().resolve(), here, *here.parents]
    seen: set[Path] = set()
    for start in starts:
        for candidate in [start, *start.parents]:
            if candidate in seen:
                continue
            seen.add(candidate)
            package = candidate / "src" / "nndl_project" / "__init__.py"
            if (candidate / "pyproject.toml").is_file() and package.is_file():
                return candidate
    raise FileNotFoundError("Could not find the project root from the working directory or this file.")


def split_paths(split: str) -> tuple[Path, Path]:
    """Return the wav directory and the strong-label tsv for one synthetic split.

    Only synthetic21_train and synthetic21_validation are valid. Public eval
    and the dcase2019 condition folders are not reachable through this function.
    """
    if split not in {"train", "validation"}:
        raise ValueError("split must be 'train' or 'validation'")
    root = project_root() / "data" / "raw" / "desed" / "dcase_synth"
    if split == "train":
        audio = root / "audio" / "train" / "synthetic21_train" / "soundscapes"
        tsv = root / "metadata" / "train" / "synthetic21_train" / "soundscapes.tsv"
    else:
        audio = root / "audio" / "validation" / "synthetic21_validation" / "soundscapes"
        tsv = root / "metadata" / "validation" / "synthetic21_validation" / "soundscapes.tsv"
    return audio, tsv


def clip_path(audio_dir: Path, filename: str) -> Path:
    """Resolve one wav name inside audio_dir and reject paths that leave it."""
    if not isinstance(filename, str) or filename == "":
        raise ValueError("filename must be a plain .wav name")
    if filename != Path(filename).name:
        raise ValueError("filename must not contain a directory")
    if any(part in filename for part in ("/", "\\", ":", "..")):
        raise ValueError("filename must not contain a path separator")
    if not filename.endswith(".wav") or filename.startswith("._"):
        raise ValueError("filename must be a .wav name")
    root = audio_dir.resolve()
    path = (root / filename).resolve()
    if path.parent != root:
        raise ValueError("resolved clip path left the requested audio directory")
    return path


def list_clip_names(audio_dir: Path) -> list[str]:
    """List wav names in numeric order. Jams and txt copies are ignored."""
    names: list[str] = []
    with os.scandir(audio_dir) as entries:
        for entry in entries:
            name = entry.name
            if not entry.is_file() or not name.endswith(".wav") or name.startswith("._"):
                continue
            names.append(name)

    def sort_key(name: str) -> tuple[int, int | str]:
        stem = name[:-4]
        if stem.isdigit():
            return (0, int(stem))
        return (1, stem)

    names.sort(key=sort_key)
    return names


def read_strong_tsv(tsv_path: Path) -> dict[str, list[Event]]:
    """Read filename, onset, offset, event_label. Group rows by filename."""
    grouped: dict[str, list[Event]] = {}
    with tsv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = list(reader.fieldnames or [])
        expected = ["filename", "onset", "offset", "event_label"]
        if fieldnames != expected:
            raise ValueError(f"{tsv_path} columns are {fieldnames}, expected {expected}")
        for line_number, row in enumerate(reader, start=2):
            filename = row["filename"]
            try:
                _require_plain_wav_name(filename)
                onset = float(row["onset"])
                offset = float(row["offset"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{tsv_path} line {line_number} is not a usable strong label") from exc
            if not np.isfinite(onset) or not np.isfinite(offset) or offset <= onset or onset < 0:
                raise ValueError(f"{tsv_path} line {line_number} has onset {onset} and offset {offset}")
            label = row["event_label"]
            if label not in CLASSES:
                raise ValueError(f"{tsv_path} line {line_number} has unknown label {label}")
            grouped.setdefault(filename, []).append(Event(label, onset, offset))
    return grouped


def _require_plain_wav_name(filename: str) -> None:
    if filename != Path(filename).name or any(part in filename for part in ("/", "\\", ":", "..")):
        raise ValueError(f"label filename is not a plain wav name: {filename}")
    if not filename.endswith(".wav"):
        raise ValueError(f"label filename is not a wav name: {filename}")


def events_for_split(split: str) -> dict[str, list[Event]]:
    """Load strong labels for one split. The other split is not read."""
    _audio, tsv = split_paths(split)
    return read_strong_tsv(tsv)


def frame_targets(events: list[Event], n_frames: int = N_FRAMES) -> np.ndarray:
    """Mark a frame positive when its analysis window overlaps a strong label.

    Frame i covers samples [i * hop, i * hop + n_fft). Overlap is used so an
    event shorter than the hop is not dropped between frame centers.
    """
    if n_frames != N_FRAMES:
        raise ValueError(f"this baseline expects {N_FRAMES} frames, got {n_frames}")
    targets = np.zeros((n_frames, len(CLASSES)), dtype=np.uint8)
    if not events:
        return targets
    class_index = {name: index for index, name in enumerate(CLASSES)}
    starts = np.arange(n_frames, dtype=np.float64) * (HOP / SAMPLE_RATE)
    ends = starts + (N_FFT / SAMPLE_RATE)
    for event in events:
        if event.label not in class_index:
            raise ValueError(f"unknown label {event.label}")
        column = class_index[event.label]
        hit = (event.onset < ends) & (event.offset > starts)
        targets[hit, column] = 1
    return targets
