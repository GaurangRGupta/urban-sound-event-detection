"""Train and validation share wav names. Those clips must stay apart."""

from pathlib import Path

import pytest

from nndl_project.data import clip_path, read_strong_tsv, split_paths
from nndl_project.features import load_mono_16k
from nndl_project.constants import TARGET_SAMPLES


def test_split_names_reject_the_held_out_sets() -> None:
    with pytest.raises(ValueError):
        split_paths("public")
    with pytest.raises(ValueError):
        split_paths("dcase2019")


def test_clip_path_rejects_paths_that_leave_the_folder() -> None:
    audio_dir, _tsv = split_paths("train")
    rejected = [
        "../0.wav",
        "..\\0.wav",
        "nested/0.wav",
        "C:/Windows/0.wav",
        "..",
        "0.wav/../1.wav",
        "",
        "0.wav ",
    ]
    for filename in rejected:
        with pytest.raises(ValueError):
            clip_path(audio_dir, filename)


def test_same_filename_is_a_different_clip_in_each_split() -> None:
    train_audio, train_tsv = split_paths("train")
    val_audio, val_tsv = split_paths("validation")
    if not (train_audio / "0.wav").is_file() or not (val_audio / "0.wav").is_file():
        pytest.skip("synthetic clips are not on disk")
    train_wav = clip_path(train_audio, "0.wav")
    val_wav = clip_path(val_audio, "0.wav")
    assert train_wav != val_wav
    assert train_wav.parent == train_audio.resolve()
    assert val_wav.parent == val_audio.resolve()
    assert "dcase2019" not in train_wav.parts
    assert "public" not in train_wav.parts
    assert "dcase2019" not in val_wav.parts
    train_events = [(event.label, event.onset, event.offset) for event in read_strong_tsv(train_tsv)["0.wav"]]
    val_events = [(event.label, event.onset, event.offset) for event in read_strong_tsv(val_tsv)["0.wav"]]
    assert train_events
    assert val_events
    assert train_events != val_events
    waveform = load_mono_16k(train_wav)
    assert waveform.shape == (TARGET_SAMPLES,)
    assert waveform.dtype.name == "float32"
    assert Path(val_wav).stat().st_size > 0
