"""16 kHz mono log-mel frames. The matrix is a loudness view, not an image."""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from nndl_project.constants import (
    FMAX,
    FMIN,
    HOP,
    LENGTH_TOLERANCE_SECONDS,
    LOG_FLOOR,
    N_FFT,
    N_FRAMES,
    N_MELS,
    SAMPLE_RATE,
    TARGET_SAMPLES,
)
from nndl_project.data import Event, clip_path, frame_targets


def load_mono_16k(path: Path) -> np.ndarray:
    """Read one clip, mix to one channel, and resample to 16 kHz and 10 seconds."""
    audio, source_rate = sf.read(path, always_2d=True, dtype="float32")
    waveform = audio.mean(axis=1) if audio.shape[1] != 1 else audio[:, 0]
    source_rate = int(source_rate)
    if source_rate != SAMPLE_RATE:
        divisor = math.gcd(source_rate, SAMPLE_RATE)
        waveform = signal.resample_poly(waveform, SAMPLE_RATE // divisor, source_rate // divisor)
    waveform = np.asarray(waveform, dtype=np.float32)
    difference = len(waveform) - TARGET_SAMPLES
    tolerance = int(LENGTH_TOLERANCE_SECONDS * SAMPLE_RATE)
    if abs(difference) > tolerance:
        raise ValueError(
            f"{path.name} is {len(waveform) / SAMPLE_RATE:.3f} s at {SAMPLE_RATE} Hz, expected 10 s"
        )
    if difference > 0:
        waveform = waveform[:TARGET_SAMPLES]
    elif difference < 0:
        waveform = np.pad(waveform, (0, -difference))
    return np.ascontiguousarray(waveform, dtype=np.float32)


@lru_cache(maxsize=1)
def mel_filterbank() -> np.ndarray:
    """HTK mel filters, shape (n_mels, n_fft // 2 + 1), each scaled to unit area."""

    def hz_to_mel(hz: np.ndarray) -> np.ndarray:
        return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)

    def mel_to_hz(mel: np.ndarray) -> np.ndarray:
        return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)

    n_freqs = N_FFT // 2 + 1
    mel_points = np.linspace(hz_to_mel(np.array(FMIN)), hz_to_mel(np.array(FMAX)), N_MELS + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((N_FFT + 1) * hz_points / SAMPLE_RATE).astype(int)
    bins = np.clip(bins, 0, n_freqs - 1)
    weights = np.zeros((N_MELS, n_freqs), dtype=np.float64)
    for band in range(N_MELS):
        left = int(bins[band])
        center = int(bins[band + 1])
        right = int(bins[band + 2])
        if center <= left:
            center = left + 1
        if right <= center:
            right = center + 1
        right = min(right, n_freqs)
        center = min(center, right - 1)
        up = np.arange(left, center, dtype=np.float64)
        down = np.arange(center, right, dtype=np.float64)
        if up.size:
            weights[band, left:center] = (up - left) / (center - left)
        if down.size:
            weights[band, center:right] = (right - down) / (right - center)
    span = hz_points[2:] - hz_points[:N_MELS]
    weights *= (2.0 / np.maximum(span, 1e-6))[:, np.newaxis]
    if np.any(weights.sum(axis=1) <= 0):
        raise RuntimeError("a mel filter is empty")
    return weights.astype(np.float32)


def extract_named_clip(audio_dir: str, filename: str, events: list[Event]) -> tuple[np.ndarray, np.ndarray]:
    """Load one wav from the given split folder and return features and targets.

    The filename is resolved only inside audio_dir, so a shared name such as
    0.wav cannot pull the clip from the other split.
    """
    waveform = load_mono_16k(clip_path(Path(audio_dir), filename))
    features = log_mel(waveform)
    return features, frame_targets(events, features.shape[0])


def log_mel(waveform: np.ndarray) -> np.ndarray:
    """Return log mel power with shape (N_FRAMES, N_MELS), float32."""
    if waveform.shape != (TARGET_SAMPLES,):
        raise ValueError(f"expected {(TARGET_SAMPLES,)} samples, got {waveform.shape}")
    _frequencies, _times, spectrum = signal.stft(
        waveform,
        fs=SAMPLE_RATE,
        window="hann",
        nperseg=N_FFT,
        noverlap=N_FFT - HOP,
        nfft=N_FFT,
        boundary=None,
        padded=False,
        return_onesided=True,
    )
    power = spectrum.real.astype(np.float32) ** 2 + spectrum.imag.astype(np.float32) ** 2
    mel = mel_filterbank() @ power
    logged = np.log(np.maximum(mel, LOG_FLOOR)).T.astype(np.float32)
    if logged.shape != (N_FRAMES, N_MELS):
        raise RuntimeError(f"log-mel shape is {logged.shape}, expected {(N_FRAMES, N_MELS)}")
    if not np.isfinite(logged).all():
        raise RuntimeError("log-mel contains a non-finite value")
    return np.ascontiguousarray(logged)
