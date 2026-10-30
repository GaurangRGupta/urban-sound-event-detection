"""Run a trained model on one or more wav files and print the detected events.

This is the project output described in the problem statement: for a clip,
a list of events, each with a class name, a start time in seconds, an end
time in seconds, and a score between 0 and 1.

Usage, from the project directory:

    uv run python -m nndl_project.predict path/to/clip.wav
    uv run python -m nndl_project.predict a.wav b.wav --street --json events.json

The default checkpoint is the CRNN, the best model on validation. A frame
GRU checkpoint also loads. `--street` relabels the events with the fixed
street classes from `label_map.py`. `--thresholds` reads per-class
thresholds written by `nndl_project.evaluate`. Without it every class uses
0.5, the setting all the validation tables were scored with.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from nndl_project.constants import CLASSES, CLIP_SECONDS, MEDIAN_FRAMES, N_MELS, THRESHOLD
from nndl_project.crnn import SmallCRNN
from nndl_project.data import Event, project_root
from nndl_project.decode import class_thresholds, decode_events
from nndl_project.features import load_any_16k, log_mel
from nndl_project.label_map import to_street_events
from nndl_project.sequence import FrameGRU


@dataclass
class LoadedModel:
    """A network in eval mode plus the standardization it was trained with."""

    name: str
    network: nn.Module
    mean: np.ndarray
    std: np.ndarray
    epoch: int

    def probabilities(self, features: np.ndarray) -> np.ndarray:
        """Frame probabilities for a batch of log-mel matrices.

        features has shape (batch, frames, bands) or (frames, bands). The
        result has the same leading shape and one column per DESED class.
        """
        single = features.ndim == 2
        batch = features[np.newaxis] if single else features
        normalized = ((batch - self.mean) / self.std).astype(np.float32, copy=False)
        with torch.no_grad():
            logits = self.network(torch.from_numpy(np.ascontiguousarray(normalized)))
            probs = torch.sigmoid(logits).numpy().astype(np.float32)
        return probs[0] if single else probs


def default_checkpoint(model: str = "crnn") -> Path:
    folder = {"crnn": "crnn", "frame_gru": "frame_gru"}[model]
    return project_root() / "data" / "cache" / folder / "full" / "model_last.pt"


def load_model(checkpoint: Path) -> LoadedModel:
    """Load a CRNN or frame GRU checkpoint written by the training scripts."""
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"No checkpoint at {checkpoint}. Train first, for example "
            "`uv run python -m nndl_project.train_crnn`, or pull data/cache with DVC."
        )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    identity = payload.get("identity", {})
    kind = identity.get("model", "")
    hidden = int(identity.get("hidden", 64))
    if kind == "small_crnn":
        network: nn.Module = SmallCRNN(channels=int(identity.get("channels", 16)), hidden=hidden)
    elif "gru" in str(kind):
        network = FrameGRU(hidden=hidden)
    else:
        raise ValueError(f"{checkpoint} holds model {kind!r}. Only the CRNN and the frame GRU load here.")
    network.load_state_dict(payload["model_state"])
    network.eval()
    mean = np.asarray(payload["mean"], dtype=np.float32).reshape(N_MELS)
    std = np.asarray(payload["std"], dtype=np.float32).reshape(N_MELS)
    return LoadedModel(str(kind), network, mean, std, int(payload.get("epoch", 0)))


def clip_features(path: Path) -> np.ndarray:
    """Any wav, any channel count, any sample rate, as a (497, 64) log-mel.

    A clip longer than 10 s is cut to its first 10 s. A shorter clip is padded.
    """
    waveform, _seconds = load_any_16k(path)
    return log_mel(waveform)


def predict_clip(
    model: LoadedModel,
    path: Path,
    *,
    threshold: float | list[float] = THRESHOLD,
    median_frames: int = MEDIAN_FRAMES,
    street: bool = False,
) -> list[Event]:
    """Return the events for one clip, sorted by start time.

    For a clip shorter than 10 s, events in the silent padding are dropped
    and an event that runs into the padding ends at the real end of the clip.
    """
    waveform, seconds = load_any_16k(path)
    probabilities = model.probabilities(log_mel(waveform))
    events = decode_events(probabilities, threshold=threshold, median_frames=median_frames)
    end = min(seconds, CLIP_SECONDS)
    events = [Event(e.label, e.onset, min(e.offset, end), e.score) for e in events if e.onset < end]
    if street:
        events = to_street_events(events)
    return sorted(events, key=lambda e: (e.onset, e.label))


def events_as_dicts(events: list[Event]) -> list[dict[str, object]]:
    return [
        {
            "label": e.label,
            "start": round(e.onset, 3),
            "end": round(e.offset, 3),
            "score": None if e.score is None else round(float(e.score), 3),
        }
        for e in events
    ]


def read_thresholds(path: Path) -> tuple[list[float], int]:
    """Read per-class thresholds and the median length chosen by evaluate.py."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    tuned = payload["tuned_decode"]
    values = [float(tuned["thresholds"][name]) for name in CLASSES]
    class_thresholds(values)
    return values, int(tuned["median_frames"])


def run(argv: list[str] | None = None) -> list[dict[str, object]]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("wavs", nargs="+", type=Path, help="one or more wav files")
    parser.add_argument("--model", choices=["crnn", "frame_gru"], default="crnn")
    parser.add_argument("--checkpoint", type=Path, default=None, help="overrides --model")
    parser.add_argument("--thresholds", type=Path, default=None, help="reports/final_evaluation.json")
    parser.add_argument("--street", action="store_true", help="report the ten street class names")
    parser.add_argument("--json", type=Path, default=None, help="also write the events to this file")
    args = parser.parse_args(argv)

    checkpoint = args.checkpoint or default_checkpoint(args.model)
    model = load_model(checkpoint)
    threshold: float | list[float] = THRESHOLD
    median = MEDIAN_FRAMES
    if args.thresholds is not None:
        threshold, median = read_thresholds(args.thresholds)

    results: list[dict[str, object]] = []
    for wav in args.wavs:
        events = predict_clip(model, wav, threshold=threshold, median_frames=median, street=args.street)
        results.append({"file": str(wav), "events": events_as_dicts(events)})
        print(f"{wav}  ({len(events)} events, model {model.name} epoch {model.epoch})")
        if not events:
            print("  no event above threshold")
        for e in events:
            print(f"  {e.label:<28} {e.onset:7.3f} s -> {e.offset:7.3f} s   score {e.score:.3f}")
    if args.json is not None:
        args.json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {args.json}")
    return results


def main() -> None:
    try:
        run()
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
