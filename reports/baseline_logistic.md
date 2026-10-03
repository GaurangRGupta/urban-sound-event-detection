# Frame-wise logistic regression baseline

These scores are the frame-wise logistic regression baseline on all synthetic21_validation clips. The threshold was not tuned on this split.

Command: `uv run python -m nndl_project.baseline`

The classifier is one logistic regression per DESED class. Each input row is one log-mel frame. This is not a convolutional network, and the log-mel matrix is not treated as a picture.

## Setup

- Train audio and labels: synthetic21_train only, 10000 clips, 0.wav through 9999.wav.
- Validation audio and labels: synthetic21_validation only, 2500 clips, 0.wav through 2499.wav.
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
- Elapsed time: 715.6 seconds.

Positive training frames, out of 4970000: Alarm_bell_ringing 149447 (3.01 percent), Blender 213686 (4.30 percent), Cat 130789 (2.63 percent), Dishes 157369 (3.17 percent), Dog 135712 (2.73 percent), Electric_shaver_toothbrush 473184 (9.52 percent), Frying 629952 (12.68 percent), Running_water 407436 (8.20 percent), Speech 1271431 (25.58 percent), Vacuum_cleaner 492193 (9.90 percent).

## Output of one clip

Each detected event has a class name, a start time in seconds, an end time in seconds, and a score from 0 to 1. Overlapping classes are separate events. The first validation clip in this run is 0.wav.

Reference labels:

- Dog from 5.050 s to 5.310 s
- Speech from 6.832 s to 8.674 s
- Dog from 7.412 s to 7.797 s

Baseline events:

- Cat from 3.580 s to 3.660 s, score 0.503
- Cat from 5.100 s to 5.260 s, score 0.513
- Cat from 7.000 s to 7.140 s, score 0.506
- Cat from 7.400 s to 7.520 s, score 0.503
- Cat from 7.940 s to 8.100 s, score 0.512
- Cat from 8.800 s to 8.900 s, score 0.501
- Dishes from 8.240 s to 8.320 s, score 0.507
- Dog from 3.580 s to 3.660 s, score 0.502
- Dog from 4.600 s to 4.700 s, score 0.520
- Dog from 4.960 s to 5.160 s, score 0.604
- Dog from 5.300 s to 5.440 s, score 0.560
- Dog from 6.960 s to 7.160 s, score 0.522
- Dog from 7.380 s to 7.640 s, score 0.535
- Dog from 9.420 s to 9.820 s, score 0.635
- Frying from 8.180 s to 8.360 s, score 0.635
- Speech from 2.900 s to 3.060 s, score 0.664
- Speech from 3.500 s to 3.860 s, score 0.617
- Speech from 4.060 s to 4.240 s, score 0.732
- Speech from 4.560 s to 5.760 s, score 0.739
- Speech from 6.000 s to 6.260 s, score 0.685
- Speech from 6.440 s to 8.720 s, score 0.770
- Speech from 8.800 s to 8.960 s, score 0.575
- Speech from 9.180 s to 9.360 s, score 0.525
- Speech from 9.460 s to 9.680 s, score 0.559

## Segment scores

| class | precision | recall | f1 | tp | fp | fn |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Alarm_bell_ringing | 0.130 | 0.088 | 0.105 | 196 | 1311 | 2043 |
| Blender | 0.108 | 0.212 | 0.143 | 510 | 4222 | 1895 |
| Cat | 0.188 | 0.370 | 0.249 | 503 | 2177 | 856 |
| Dishes | 0.116 | 0.587 | 0.194 | 1097 | 8332 | 771 |
| Dog | 0.133 | 0.853 | 0.230 | 1214 | 7924 | 210 |
| Electric_shaver_toothbrush | 0.186 | 0.768 | 0.300 | 1612 | 7039 | 487 |
| Frying | 0.269 | 0.793 | 0.401 | 2781 | 7564 | 728 |
| Running_water | 0.152 | 0.671 | 0.248 | 1642 | 9150 | 805 |
| Speech | 0.675 | 0.918 | 0.778 | 12217 | 5883 | 1088 |
| Vacuum_cleaner | 0.152 | 0.588 | 0.242 | 1475 | 8215 | 1035 |
| micro | 0.273 | 0.701 | 0.393 | 23247 | 61817 | 9918 |
| macro f1 |  |  | 0.289 |  |  |  |

## Event scores

| class | precision | recall | f1 | tp | fp | fn |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Alarm_bell_ringing | 0.000 | 0.000 | 0.000 | 0 | 1498 | 431 |
| Blender | 0.000 | 0.000 | 0.000 | 0 | 4988 | 266 |
| Cat | 0.005 | 0.035 | 0.009 | 15 | 2847 | 414 |
| Dishes | 0.029 | 0.150 | 0.049 | 197 | 6529 | 1112 |
| Dog | 0.024 | 0.353 | 0.044 | 194 | 8054 | 356 |
| Electric_shaver_toothbrush | 0.010 | 0.203 | 0.018 | 58 | 5936 | 228 |
| Frying | 0.015 | 0.268 | 0.029 | 101 | 6569 | 276 |
| Running_water | 0.002 | 0.056 | 0.004 | 17 | 8829 | 289 |
| Speech | 0.102 | 0.258 | 0.146 | 1013 | 8939 | 2914 |
| Vacuum_cleaner | 0.002 | 0.068 | 0.004 | 17 | 8961 | 234 |
| micro | 0.025 | 0.198 | 0.044 | 1612 | 63150 | 6520 |
| macro f1 |  |  | 0.030 |  |  |  |

A class row is n/a when that class has no reference events and no predictions in the scored clips. Macro F1 averages the classes that have a defined F1. Micro F1 pools tp, fp, and fn across classes.
