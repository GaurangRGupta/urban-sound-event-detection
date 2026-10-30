"""The prediction command turns any wav into a list of timed, scored events."""

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

torch = pytest.importorskip("torch")

from nndl_project.constants import CLASSES, N_MELS  # noqa: E402
from nndl_project.crnn import SmallCRNN  # noqa: E402
from nndl_project.label_map import STREET_CLASSES  # noqa: E402
from nndl_project.predict import load_model, run  # noqa: E402


def _checkpoint(path: Path, bias: float) -> Path:
    torch.manual_seed(0)
    model = SmallCRNN()
    with torch.no_grad():
        # A large positive bias on Speech and Dishes makes those classes fire.
        model.head.bias.fill_(-bias)
        model.head.bias[CLASSES.index("Speech")] = bias
        model.head.bias[CLASSES.index("Dishes")] = bias
    torch.save(
        {
            "epoch": 3,
            "model_state": model.state_dict(),
            "mean": np.zeros(N_MELS, dtype=np.float32),
            "std": np.ones(N_MELS, dtype=np.float32),
            "identity": {"model": "small_crnn", "channels": 16, "hidden": 64},
        },
        path,
    )
    return path


def _wav(path: Path, seconds: float, channels: int, rate: int) -> Path:
    rng = np.random.default_rng(0)
    audio = 0.05 * rng.standard_normal((int(seconds * rate), channels)).astype(np.float32)
    sf.write(path, audio, rate)
    return path


def test_checkpoint_loads_in_eval_mode(tmp_path: Path) -> None:
    model = load_model(_checkpoint(tmp_path / "m.pt", 8.0))
    assert model.name == "small_crnn"
    assert model.epoch == 3
    assert not model.network.training


def test_events_have_label_start_end_and_score(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "m.pt", 8.0)
    # A short stereo clip at 44.1 kHz: the loader must mix, resample, and pad.
    wav = _wav(tmp_path / "clip.wav", 9.3, 2, 44_100)
    out = tmp_path / "events.json"
    results = run([str(wav), "--checkpoint", str(checkpoint), "--json", str(out)])
    events = results[0]["events"]
    assert {e["label"] for e in events} == {"Speech", "Dishes"}
    for e in events:
        # The clip is 9.3 s long, so nothing may run into the padding.
        assert 0.0 <= e["start"] < e["end"] <= 9.3
        assert 0.0 <= e["score"] <= 1.0
    assert json.loads(out.read_text())[0]["events"] == events


def test_street_output_uses_street_names(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "m.pt", 8.0)
    wav = _wav(tmp_path / "clip.wav", 10.0, 1, 16_000)
    results = run([str(wav), "--checkpoint", str(checkpoint), "--street"])
    labels = {e["label"] for e in results[0]["events"]}
    assert labels == {"speech_crowd", "other"}
    assert labels <= set(STREET_CLASSES)


def test_missing_checkpoint_is_a_clear_error(tmp_path: Path) -> None:
    wav = _wav(tmp_path / "clip.wav", 10.0, 1, 16_000)
    with pytest.raises(FileNotFoundError):
        run([str(wav), "--checkpoint", str(tmp_path / "nothing.pt")])
