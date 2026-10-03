"""Fixed settings for the first baseline.

The class names are the ten DESED labels. This baseline does not collapse
them onto the street taxonomy. The feature settings are part of the cache
key, so a change here must not silently reuse old arrays.
"""

from __future__ import annotations

CLASSES: tuple[str, ...] = (
    "Alarm_bell_ringing",
    "Blender",
    "Cat",
    "Dishes",
    "Dog",
    "Electric_shaver_toothbrush",
    "Frying",
    "Running_water",
    "Speech",
    "Vacuum_cleaner",
)

SAMPLE_RATE = 16_000
CLIP_SECONDS = 10.0
SOURCE_SAMPLE_RATE = 44_100
N_FFT = 1024
HOP = 320
N_MELS = 64
FMIN = 0.0
FMAX = 8_000.0
LOG_FLOOR = 1e-10

# 10 s at 16 kHz, then one frame per hop that still fits a full window.
TARGET_SAMPLES = int(SAMPLE_RATE * CLIP_SECONDS)
N_FRAMES = 1 + (TARGET_SAMPLES - N_FFT) // HOP

MEDIAN_FRAMES = 11
THRESHOLD = 0.5
SEGMENT_SECONDS = 1.0
ONSET_COLLAR_SECONDS = 0.2
OFFSET_COLLAR_SECONDS = 0.2
OFFSET_COLLAR_FRACTION = 0.2

FEATURE_VERSION = 1
LENGTH_TOLERANCE_SECONDS = 0.05
