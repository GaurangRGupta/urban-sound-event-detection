"""Load the four trained models for the presentation page.

The numbers in SCOREBOARD are the published validation scores. They are not
recomputed here. Live detection uses the same fixed decode as those tables:
median filter of 11 frames, then threshold 0.5.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from nndl_project.constants import CLASSES, CLIP_SECONDS, MEDIAN_FRAMES, N_MELS, THRESHOLD
from nndl_project.context import ContextMLP, n_inputs, stack_context
from nndl_project.crnn import SmallCRNN
from nndl_project.data import Event, project_root
from nndl_project.decode import decode_events
from nndl_project.features import log_mel
from nndl_project.label_map import to_street_events
from nndl_project.sequence import FrameGRU

# Segment micro, segment macro, event micro, event macro. All 2500
# synthetic21_validation clips. Fixed decode, not a tuned threshold.
SCOREBOARD: tuple[dict[str, object], ...] = (
    {
        "model": "Logistic baseline",
        "detail": "one frame at a time, 5 passes",
        "segment_micro": 0.393,
        "segment_macro": 0.289,
        "event_micro": 0.044,
        "event_macro": 0.030,
    },
    {
        "model": "Context MLP",
        "detail": "11-frame window, epoch 432",
        "segment_micro": 0.398,
        "segment_macro": 0.349,
        "event_micro": 0.053,
        "event_macro": 0.041,
    },
    {
        "model": "Frame GRU",
        "detail": "whole clip, epoch 30",
        "segment_micro": 0.509,
        "segment_macro": 0.438,
        "event_micro": 0.104,
        "event_macro": 0.096,
    },
    {
        "model": "CRNN",
        "detail": "two convolutions plus GRU, epoch 10",
        "segment_micro": 0.612,
        "segment_macro": 0.542,
        "event_micro": 0.142,
        "event_macro": 0.155,
    },
)

# Event F1 by DESED class, same order as SCOREBOARD.
CLASS_EVENT_F1: dict[str, tuple[float, float, float, float]] = {
    "Alarm_bell_ringing": (0.000, 0.017, 0.014, 0.119),
    "Blender": (0.000, 0.009, 0.040, 0.128),
    "Cat": (0.009, 0.019, 0.042, 0.045),
    "Dishes": (0.049, 0.061, 0.047, 0.069),
    "Dog": (0.044, 0.035, 0.025, 0.051),
    "Electric_shaver_toothbrush": (0.018, 0.017, 0.146, 0.198),
    "Frying": (0.029, 0.024, 0.194, 0.358),
    "Running_water": (0.004, 0.006, 0.060, 0.154),
    "Speech": (0.146, 0.212, 0.205, 0.185),
    "Vacuum_cleaner": (0.004, 0.006, 0.186, 0.244),
}


def scores_payload() -> dict[str, object]:
    """JSON for the scoreboard. The live page does not recompute these."""
    return {
        "set": "synthetic21_validation, 2500 clips",
        "decode": "median 11 frames, threshold 0.5",
        "models": [row["model"] for row in SCOREBOARD],
        "rows": list(SCOREBOARD),
        "class_event_f1": {name: list(values) for name, values in CLASS_EVENT_F1.items()},
        "held_out": "The real public-test score has not been run. This table is the synthetic validation set.",
    }


@dataclass
class DemoModel:
    """One trained model that turns a log-mel matrix into frame probabilities."""

    name: str
    detail: str
    mean: np.ndarray
    std: np.ndarray
    kind: str
    network: nn.Module | None = None
    logistic: object | None = None
    radius: int = 5

    def probabilities(self, features: np.ndarray) -> np.ndarray:
        """features is one clip, shape (frames, bands), before standardization."""
        standardized = ((features - self.mean) / self.std).astype(np.float32, copy=False)
        if self.kind == "logistic":
            design = np.ascontiguousarray(standardized.reshape(-1, N_MELS))
            columns = [model.predict_proba(design)[:, 1] for model in self.logistic]
            return np.stack(columns, axis=1).astype(np.float32)
        if self.kind == "mlp":
            window = stack_context(standardized, self.radius)
            with torch.no_grad():
                logits = self.network(torch.from_numpy(np.ascontiguousarray(window)))
                return torch.sigmoid(logits).numpy().astype(np.float32)
        with torch.no_grad():
            logits = self.network(torch.from_numpy(standardized[None, :, :]))
            return torch.sigmoid(logits)[0].numpy().astype(np.float32)


def _torch_mean_std(payload: dict) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(payload["mean"], dtype=np.float32).reshape(N_MELS)
    std = np.asarray(payload["std"], dtype=np.float32).reshape(N_MELS)
    return mean, std


def load_logistic(path: Path) -> DemoModel:
    import joblib

    payload = joblib.load(path)
    mean = np.asarray(payload["mean"], dtype=np.float32).reshape(N_MELS)
    std = np.asarray(payload["std"], dtype=np.float32).reshape(N_MELS)
    return DemoModel(
        "Logistic baseline",
        f"{int(payload.get('epochs', 5))} passes over the training clips",
        mean,
        std,
        "logistic",
        logistic=payload["models"],
    )


def load_mlp(path: Path) -> DemoModel:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    identity = payload.get("identity") or {}
    radius = int(identity.get("context_radius", 5))
    hidden = int(identity.get("hidden", 128))
    network = ContextMLP(n_inputs(radius), hidden, len(CLASSES))
    network.load_state_dict(payload["model_state"])
    network.eval()
    mean, std = _torch_mean_std(payload)
    return DemoModel(
        "Context MLP",
        f"epoch {int(payload.get('epoch', 0))}",
        mean,
        std,
        "mlp",
        network=network,
        radius=radius,
    )


def load_gru(path: Path) -> DemoModel:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    identity = payload.get("identity") or {}
    hidden = int(identity.get("hidden", 64))
    network = FrameGRU(hidden=hidden)
    network.load_state_dict(payload["model_state"])
    network.eval()
    mean, std = _torch_mean_std(payload)
    return DemoModel(
        "Frame GRU",
        f"epoch {int(payload.get('epoch', 0))}",
        mean,
        std,
        "gru",
        network=network,
    )


def load_crnn(path: Path) -> DemoModel:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    identity = payload.get("identity") or {}
    network = SmallCRNN(channels=int(identity.get("channels", 16)), hidden=int(identity.get("hidden", 64)))
    network.load_state_dict(payload["model_state"])
    network.eval()
    mean, std = _torch_mean_std(payload)
    return DemoModel(
        "CRNN",
        f"epoch {int(payload.get('epoch', 0))}",
        mean,
        std,
        "crnn",
        network=network,
    )


def demo_checkpoints() -> list[tuple[str, Path, object]]:
    root = project_root() / "data" / "cache"
    return [
        ("Logistic baseline", root / "logmel_v1" / "train_10000" / "models.joblib", load_logistic),
        ("Context MLP", root / "context_mlp" / "full" / "model_last.pt", load_mlp),
        ("Frame GRU", root / "frame_gru" / "full" / "model_last.pt", load_gru),
        ("CRNN", root / "crnn" / "full" / "model_last.pt", load_crnn),
    ]


def load_demo_models() -> list[DemoModel]:
    """Load every checkpoint that is on this machine. Missing ones are skipped."""
    loaded: list[DemoModel] = []
    for name, path, loader in demo_checkpoints():
        if not path.is_file():
            print(f"Missing {name}: {path}")
            continue
        print(f"Loading {name} from {path}")
        loaded.append(loader(path))
    return loaded


def events_for_features(
    model: DemoModel,
    features: np.ndarray,
    seconds: float,
    *,
    street: bool,
) -> list[Event]:
    """Decode one clip with the fixed threshold. Short clips drop the padding."""
    probabilities = model.probabilities(features)
    events = decode_events(probabilities, threshold=THRESHOLD, median_frames=MEDIAN_FRAMES)
    end = min(seconds, CLIP_SECONDS)
    events = [Event(event.label, event.onset, min(event.offset, end), event.score) for event in events if event.onset < end]
    if street:
        events = to_street_events(events)
    return sorted(events, key=lambda event: (event.onset, event.label))


def example_wav() -> Path:
    """Validation clip 2150.wav.

    The CRNN matches all four reference events here and adds none.
    Clip 0.wav finds the sounds but misses the 200 ms collar.
    """
    return (
        project_root()
        / "data"
        / "raw"
        / "desed"
        / "dcase_synth"
        / "audio"
        / "validation"
        / "synthetic21_validation"
        / "soundscapes"
        / "2150.wav"
    )
