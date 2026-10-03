# Frame-wise logistic regression baseline

This file is a pipeline check on a prefix of the synthetic clips. It is not the full-data baseline score.

Command: `uv run python -m nndl_project.baseline --limit 200`

The classifier is one logistic regression per DESED class. Each input row is one log-mel frame. This is not a convolutional network, and the log-mel matrix is not treated as a picture.

## Setup

- Train audio and labels: synthetic21_train only, 200 clips, 0.wav through 199.wav.
- Validation audio and labels: synthetic21_validation only, 200 clips, 0.wav through 199.wav.
- Train and validation both use names such as 0.wav. Each label was joined to the wav in its own split folder.
- Clips are 10 seconds. The loader mixes to one channel and resamples 44.1 kHz to 16 kHz. There is no soundscapes_16k copy on disk.
- Log-mel features: 64 bands, FFT length 1024, hop 320 samples (20 ms), 497 frames per clip.
- A frame is positive when its FFT window overlaps a strong label. The ten DESED names are kept as they are.
- Optimizer: scikit-learn SGDClassifier with log loss, L2 alpha 0.0001, averaged weights, 5 epochs, batch of 32 clips, seed 0.
- Class imbalance uses one positive weight and one negative weight per class, from the training frame counts. Those weights are not recomputed inside a batch.
- Decode: median filter of 11 frames, then threshold 0.5. An event score is the mean smoothed probability inside the event. Boundaries use the 20 ms hop grid.
- Segment F1 uses 1 second bins. Event F1 matches one prediction to one reference when the onset is within 200 ms and the offset is within max(200 ms, 20 percent of the reference duration).
- Public eval and the dcase2019 condition folders were not used for training, features, or threshold choices.
- Package versions: scikit-learn 1.9.1, scipy 1.18.1, numpy 2.5.3.
- Elapsed time: 31.7 seconds.

Positive training frames, out of 99400: Alarm_bell_ringing 1184 (1.19 percent), Blender 2875 (2.89 percent), Cat 3650 (3.67 percent), Dishes 3565 (3.59 percent), Dog 2596 (2.61 percent), Electric_shaver_toothbrush 7028 (7.07 percent), Frying 13754 (13.84 percent), Running_water 7489 (7.53 percent), Speech 24742 (24.89 percent), Vacuum_cleaner 12348 (12.42 percent).

## Output of one clip

Each detected event has a class name, a start time in seconds, an end time in seconds, and a score from 0 to 1. Overlapping classes are separate events. The first validation clip in this run is 0.wav.

Reference labels:

- Dog from 5.050 s to 5.310 s
- Speech from 6.832 s to 8.674 s
- Dog from 7.412 s to 7.797 s

Baseline events:

- Dishes from 8.180 s to 8.360 s, score 0.954
- Dog from 5.320 s to 5.460 s, score 0.557
- Dog from 7.400 s to 7.540 s, score 0.964
- Dog from 9.360 s to 9.840 s, score 0.988
- Speech from 2.900 s to 3.080 s, score 0.808
- Speech from 3.500 s to 3.840 s, score 0.805
- Speech from 4.060 s to 4.200 s, score 0.966
- Speech from 4.560 s to 5.780 s, score 0.890
- Speech from 6.000 s to 6.260 s, score 0.818
- Speech from 6.440 s to 8.660 s, score 0.925
- Speech from 8.720 s to 8.920 s, score 0.740
- Speech from 9.180 s to 9.400 s, score 0.615
- Speech from 9.460 s to 9.740 s, score 0.737

## Segment scores

| class | precision | recall | f1 | tp | fp | fn |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Alarm_bell_ringing | 0.182 | 0.399 | 0.250 | 71 | 319 | 107 |
| Blender | 0.210 | 0.132 | 0.162 | 30 | 113 | 198 |
| Cat | 0.153 | 0.331 | 0.209 | 39 | 216 | 79 |
| Dishes | 0.103 | 0.555 | 0.174 | 86 | 749 | 69 |
| Dog | 0.101 | 0.901 | 0.182 | 82 | 728 | 9 |
| Electric_shaver_toothbrush | 0.298 | 0.574 | 0.392 | 97 | 229 | 72 |
| Frying | 0.173 | 0.360 | 0.234 | 85 | 407 | 151 |
| Running_water | 0.054 | 0.057 | 0.056 | 8 | 139 | 133 |
| Speech | 0.703 | 0.914 | 0.795 | 999 | 422 | 94 |
| Vacuum_cleaner | 0.053 | 0.005 | 0.010 | 1 | 18 | 189 |
| micro | 0.310 | 0.576 | 0.403 | 1498 | 3340 | 1101 |
| macro f1 |  |  | 0.246 |  |  |  |

## Event scores

| class | precision | recall | f1 | tp | fp | fn |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Alarm_bell_ringing | 0.003 | 0.030 | 0.005 | 1 | 391 | 32 |
| Blender | 0.000 | 0.000 | 0.000 | 0 | 154 | 25 |
| Cat | 0.004 | 0.026 | 0.007 | 1 | 245 | 37 |
| Dishes | 0.029 | 0.165 | 0.050 | 18 | 596 | 91 |
| Dog | 0.023 | 0.486 | 0.045 | 18 | 753 | 19 |
| Electric_shaver_toothbrush | 0.003 | 0.043 | 0.006 | 1 | 295 | 22 |
| Frying | 0.005 | 0.074 | 0.010 | 2 | 375 | 25 |
| Running_water | 0.000 | 0.000 | 0.000 | 0 | 155 | 20 |
| Speech | 0.077 | 0.208 | 0.112 | 65 | 780 | 248 |
| Vacuum_cleaner | 0.000 | 0.000 | 0.000 | 0 | 21 | 19 |
| micro | 0.027 | 0.165 | 0.047 | 106 | 3765 | 538 |
| macro f1 |  |  | 0.023 |  |  |  |

A class row is n/a when that class has no reference events and no predictions in the scored clips. Macro F1 averages the classes that have a defined F1. Micro F1 pools tp, fp, and fn across classes.
