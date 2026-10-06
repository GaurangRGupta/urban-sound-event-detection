# Frame gated recurrent unit

This file is a pipeline check on a prefix of the synthetic clips. It is not the full-data frame GRU score.

Command: `uv run python -m nndl_project.train_frame_gru --limit 200`

Started: 2026-10-05T14:50:10+05:30
Updated: 2026-10-05T14:59:14+05:30
Interrupted: False
Device: cpu
Project root: `C:\Users\gaura\Documents\grok_build_trial\neural_networks_and_deep_learning_project`

## Setup

- Train clips: 200, 0.wav through 199.wav.
- Validation clips: 200, 0.wav through 199.wav.
- Network: bidirectional GRU, input 64, hidden 64 in each direction, 1 layer, then linear 128 to 10 logits. Sigmoid per class, inside the loss.
- Parameters: 51210.
- Loss: binary cross-entropy on logits. Positive weight for a class is negative frames divided by positive frames: Alarm_bell_ringing 82.95, Blender 33.57, Cat 26.23, Dishes 26.88, Dog 37.29, Electric_shaver_toothbrush 13.14, Frying 6.23, Running_water 12.27, Speech 3.02, Vacuum_cleaner 7.05.
- Optimizer: Adam, learning rate 0.001, weight decay 0.0, batch 16 clips, seed 0.
- Gradient clip: global norm 5.0, applied after backward and before the Adam step. The logged grad_norm is the norm before that clip.
- Decode: median filter of 11 frames, threshold 0.5. The threshold was not chosen on validation.
- Designated model: the last finished epoch. Validation F1 is not used to pick a checkpoint.
- Feature cache version 1: 64 mel bands, FFT 1024, hop 320 at 16000 Hz, 497 frames.
- Standardization: training-set mean and standard deviation per mel band.
- PyTorch 2.14.1+cpu, NumPy 2.5.3.
- Checkpoint: `C:\Users\gaura\Documents\grok_build_trial\neural_networks_and_deep_learning_project\data\cache\frame_gru\limit200_val200\model_last.pt`.
- Log: `C:\Users\gaura\Documents\grok_build_trial\neural_networks_and_deep_learning_project\data\cache\frame_gru\limit200_val200\train.log`.
- Epochs requested: 30. Epochs finished: 30.

Logistic baseline on all 2500 validation clips: segment micro F1 0.393, segment macro F1 0.289, event micro F1 0.044, event macro F1 0.03.

Context MLP epoch 432 of 1000, interrupted, on all 2500 validation clips: segment micro F1 0.398, segment macro F1 0.349, event micro F1 0.053, event macro F1 0.041.

Compare this file with those two lines only when this run also scores all 2500 validation clips and uses the planned settings (hidden 64, learning rate 0.001, weight decay 0, gradient clip 5, batch 16, seed 0).

## First validation clip

Clip: 0.wav

Reference labels:

- Dog from 5.050 s to 5.310 s
- Speech from 6.832 s to 8.674 s
- Dog from 7.412 s to 7.797 s

Model events from the last finished epoch:

- Cat from 0.000 s to 9.240 s, score 0.708
- Dishes from 0.000 s to 0.040 s, score 0.538
- Dishes from 7.420 s to 7.680 s, score 0.591
- Dishes from 7.980 s to 8.520 s, score 0.586
- Running_water from 0.000 s to 0.040 s, score 0.654
- Running_water from 8.180 s to 8.540 s, score 0.646
- Speech from 4.640 s to 5.380 s, score 0.601
- Speech from 6.620 s to 6.800 s, score 0.514
- Speech from 6.980 s to 8.660 s, score 0.725

## Epoch history

| epoch | train weighted loss | train unweighted loss | val weighted loss | val unweighted loss | segment micro f1 | segment macro f1 | event micro f1 | event macro f1 | seconds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.2032 | 0.7020 | 1.6593 | 0.6998 | 0.276 | 0.246 | 0.035 | 0.035 | 16.5 |
| 2 | 1.1263 | 0.6931 | 1.6408 | 0.6820 | 0.287 | 0.256 | 0.040 | 0.039 | 18.0 |
| 3 | 1.0857 | 0.6659 | 1.6291 | 0.6562 | 0.298 | 0.264 | 0.043 | 0.040 | 18.7 |
| 4 | 1.0545 | 0.6324 | 1.6292 | 0.6260 | 0.312 | 0.275 | 0.047 | 0.044 | 17.9 |
| 5 | 1.0295 | 0.6080 | 1.6285 | 0.6126 | 0.319 | 0.282 | 0.046 | 0.043 | 18.7 |
| 6 | 1.0021 | 0.5858 | 1.6119 | 0.5882 | 0.335 | 0.292 | 0.050 | 0.046 | 18.0 |
| 7 | 0.9818 | 0.5639 | 1.6208 | 0.5738 | 0.341 | 0.299 | 0.051 | 0.048 | 18.0 |
| 8 | 0.9535 | 0.5496 | 1.6189 | 0.5672 | 0.350 | 0.302 | 0.053 | 0.049 | 18.1 |
| 9 | 0.9305 | 0.5379 | 1.6338 | 0.5575 | 0.352 | 0.305 | 0.059 | 0.055 | 19.6 |
| 10 | 0.9058 | 0.5184 | 1.6591 | 0.5472 | 0.359 | 0.310 | 0.062 | 0.057 | 18.4 |
| 11 | 0.8813 | 0.5134 | 1.6961 | 0.5500 | 0.359 | 0.309 | 0.060 | 0.056 | 18.1 |
| 12 | 0.8587 | 0.4980 | 1.7490 | 0.5282 | 0.366 | 0.313 | 0.063 | 0.059 | 18.0 |
| 13 | 0.8367 | 0.4742 | 1.7692 | 0.5214 | 0.372 | 0.320 | 0.069 | 0.062 | 17.8 |
| 14 | 0.8208 | 0.4807 | 1.8425 | 0.5219 | 0.367 | 0.310 | 0.059 | 0.055 | 18.5 |
| 15 | 0.7952 | 0.4557 | 1.8742 | 0.5128 | 0.374 | 0.322 | 0.070 | 0.064 | 18.5 |
| 16 | 0.7770 | 0.4572 | 1.9231 | 0.5078 | 0.381 | 0.323 | 0.062 | 0.056 | 18.2 |
| 17 | 0.7514 | 0.4469 | 2.0401 | 0.5077 | 0.380 | 0.320 | 0.061 | 0.054 | 18.1 |
| 18 | 0.7535 | 0.4364 | 2.1123 | 0.4910 | 0.390 | 0.326 | 0.064 | 0.056 | 18.6 |
| 19 | 0.7304 | 0.4303 | 2.1092 | 0.4976 | 0.382 | 0.323 | 0.067 | 0.058 | 18.1 |
| 20 | 0.6974 | 0.4046 | 2.2061 | 0.4787 | 0.392 | 0.327 | 0.067 | 0.057 | 17.9 |
| 21 | 0.6924 | 0.4146 | 2.2907 | 0.4723 | 0.393 | 0.329 | 0.064 | 0.056 | 17.8 |
| 22 | 0.6773 | 0.3983 | 2.2468 | 0.4858 | 0.397 | 0.335 | 0.069 | 0.058 | 17.9 |
| 23 | 0.6412 | 0.3911 | 2.4204 | 0.4611 | 0.402 | 0.336 | 0.066 | 0.056 | 17.7 |
| 24 | 0.6175 | 0.3694 | 2.4328 | 0.4803 | 0.399 | 0.331 | 0.061 | 0.053 | 17.8 |
| 25 | 0.6054 | 0.3676 | 2.5645 | 0.4534 | 0.402 | 0.332 | 0.069 | 0.057 | 17.9 |
| 26 | 0.5858 | 0.3520 | 2.6024 | 0.4769 | 0.397 | 0.326 | 0.068 | 0.056 | 17.8 |
| 27 | 0.5886 | 0.3401 | 2.6827 | 0.4549 | 0.401 | 0.326 | 0.073 | 0.058 | 18.0 |
| 28 | 0.5758 | 0.3374 | 2.6980 | 0.4630 | 0.408 | 0.335 | 0.067 | 0.053 | 17.9 |
| 29 | 0.5418 | 0.3359 | 2.7521 | 0.4546 | 0.406 | 0.335 | 0.070 | 0.057 | 17.8 |
| 30 | 0.5384 | 0.3273 | 3.0884 | 0.4352 | 0.414 | 0.335 | 0.075 | 0.057 | 17.8 |

## Last finished epoch

Epoch 30.

Segment scores

| class | precision | recall | f1 | tp | fp | fn |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Alarm_bell_ringing | 0.143 | 0.169 | 0.155 | 30 | 180 | 148 |
| Blender | 0.273 | 0.596 | 0.374 | 136 | 363 | 92 |
| Cat | 0.115 | 0.686 | 0.197 | 81 | 625 | 37 |
| Dishes | 0.123 | 0.452 | 0.193 | 70 | 500 | 85 |
| Dog | 0.166 | 0.824 | 0.276 | 75 | 377 | 16 |
| Electric_shaver_toothbrush | 0.347 | 0.657 | 0.454 | 111 | 209 | 58 |
| Frying | 0.332 | 0.547 | 0.413 | 129 | 260 | 107 |
| Running_water | 0.121 | 0.787 | 0.209 | 111 | 810 | 30 |
| Speech | 0.740 | 0.826 | 0.781 | 903 | 317 | 190 |
| Vacuum_cleaner | 0.188 | 0.753 | 0.301 | 143 | 617 | 47 |
| micro | 0.296 | 0.688 | 0.414 | 1789 | 4258 | 810 |
| macro f1 |  |  | 0.335 |  |  |  |

Event scores

| class | precision | recall | f1 | tp | fp | fn |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Alarm_bell_ringing | 0.000 | 0.000 | 0.000 | 0 | 157 | 33 |
| Blender | 0.044 | 0.360 | 0.078 | 9 | 197 | 16 |
| Cat | 0.014 | 0.132 | 0.025 | 5 | 364 | 33 |
| Dishes | 0.034 | 0.101 | 0.051 | 11 | 310 | 98 |
| Dog | 0.054 | 0.351 | 0.094 | 13 | 228 | 24 |
| Electric_shaver_toothbrush | 0.015 | 0.087 | 0.025 | 2 | 132 | 21 |
| Frying | 0.025 | 0.148 | 0.042 | 4 | 158 | 23 |
| Running_water | 0.024 | 0.400 | 0.045 | 8 | 328 | 12 |
| Speech | 0.131 | 0.211 | 0.162 | 66 | 438 | 247 |
| Vacuum_cleaner | 0.026 | 0.368 | 0.049 | 7 | 262 | 12 |
| micro | 0.046 | 0.194 | 0.075 | 125 | 2574 | 519 |
| macro f1 |  |  | 0.057 |  |  |  |
