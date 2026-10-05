# Context multilayer perceptron

This file is a pipeline check on a prefix of the synthetic clips. It is not the full-data context MLP score.

Command: `uv run python -m nndl_project.train_context_mlp --limit 200`

Started: 2026-10-04T00:27:09+05:30
Updated: 2026-10-04T00:27:42+05:30
Interrupted: False
Device: cpu
Project root: `C:\Users\gaura\Documents\grok_build_trial\neural_networks_and_deep_learning_project`

## Setup

- Train clips: 200, 0.wav through 199.wav.
- Validation clips: 200, 0.wav through 199.wav.
- Context: 11 frames (5 on each side), 704 inputs.
- Network: linear 704 to 128, rectified linear unit, linear 128 to 10, sigmoid per class.
- Loss: binary cross-entropy on logits. Positive weight for a class is negative frames divided by positive frames: Alarm_bell_ringing 82.95, Blender 33.57, Cat 26.23, Dishes 26.88, Dog 37.29, Electric_shaver_toothbrush 13.14, Frying 6.23, Running_water 12.27, Speech 3.02, Vacuum_cleaner 7.05.
- Optimizer: Adam, learning rate 0.001, weight decay 0.0, batch 16 clips, seed 0.
- Decode: median filter of 11 frames, threshold 0.5.
- Feature cache version 1: 64 mel bands, FFT 1024, hop 320 at 16000 Hz, 497 frames.
- PyTorch 2.14.1+cpu, NumPy 2.5.3.
- Checkpoint: `C:\Users\gaura\Documents\grok_build_trial\neural_networks_and_deep_learning_project\data\cache\context_mlp\limit200_val200\model_last.pt`.
- Log: `C:\Users\gaura\Documents\grok_build_trial\neural_networks_and_deep_learning_project\data\cache\context_mlp\limit200_val200\train.log`.
- Epochs requested: 10. Epochs finished: 10.

Logistic baseline on all 2500 validation clips, for comparison when this run is also the full split: segment micro F1 0.393, segment macro F1 0.289, event micro F1 0.044, event macro F1 0.03.

## First validation clip

Clip: 0.wav

Reference labels:

- Dog from 5.050 s to 5.310 s
- Speech from 6.832 s to 8.674 s
- Dog from 7.412 s to 7.797 s

Model events from the last finished epoch:

- Alarm_bell_ringing from 3.500 s to 3.720 s, score 0.557
- Alarm_bell_ringing from 4.060 s to 4.260 s, score 0.563
- Alarm_bell_ringing from 4.600 s to 5.400 s, score 0.591
- Alarm_bell_ringing from 6.040 s to 6.160 s, score 0.517
- Alarm_bell_ringing from 6.500 s to 7.240 s, score 0.565
- Alarm_bell_ringing from 7.380 s to 7.680 s, score 0.583
- Alarm_bell_ringing from 7.900 s to 8.500 s, score 0.524
- Alarm_bell_ringing from 8.720 s to 9.040 s, score 0.555
- Cat from 0.000 s to 0.020 s, score 0.519
- Cat from 2.800 s to 8.200 s, score 0.698
- Cat from 8.360 s to 9.880 s, score 0.658
- Dishes from 8.100 s to 8.440 s, score 0.585
- Dishes from 8.760 s to 9.060 s, score 0.563
- Dog from 3.560 s to 3.800 s, score 0.546
- Dog from 4.940 s to 5.160 s, score 0.590
- Dog from 5.340 s to 5.500 s, score 0.553
- Dog from 7.160 s to 7.700 s, score 0.617
- Dog from 9.280 s to 9.880 s, score 0.760
- Frying from 8.160 s to 8.360 s, score 0.702
- Running_water from 8.080 s to 8.460 s, score 0.708
- Speech from 2.880 s to 3.080 s, score 0.587
- Speech from 3.460 s to 3.780 s, score 0.640
- Speech from 4.020 s to 4.280 s, score 0.657
- Speech from 4.520 s to 5.680 s, score 0.722
- Speech from 5.960 s to 7.720 s, score 0.697
- Speech from 7.840 s to 9.020 s, score 0.685
- Speech from 9.140 s to 9.740 s, score 0.554

## Epoch history

| epoch | train weighted loss | train unweighted loss | val weighted loss | val unweighted loss | segment micro f1 | segment macro f1 | event micro f1 | event macro f1 | seconds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.2261 | 0.6666 | 1.6952 | 0.6267 | 0.292 | 0.264 | 0.036 | 0.032 | 3.6 |
| 2 | 1.0946 | 0.6041 | 1.7441 | 0.5896 | 0.314 | 0.275 | 0.044 | 0.040 | 3.3 |
| 3 | 1.0393 | 0.5847 | 1.6376 | 0.5787 | 0.322 | 0.280 | 0.045 | 0.041 | 3.2 |
| 4 | 1.0128 | 0.5428 | 1.7428 | 0.5531 | 0.331 | 0.285 | 0.045 | 0.039 | 3.2 |
| 5 | 0.9930 | 0.5616 | 1.7496 | 0.5590 | 0.329 | 0.286 | 0.042 | 0.037 | 3.2 |
| 6 | 0.9581 | 0.5071 | 1.7298 | 0.5382 | 0.341 | 0.287 | 0.049 | 0.042 | 3.2 |
| 7 | 0.9417 | 0.5315 | 1.7715 | 0.5543 | 0.333 | 0.293 | 0.042 | 0.035 | 3.1 |
| 8 | 0.9096 | 0.4948 | 1.8098 | 0.5173 | 0.350 | 0.292 | 0.044 | 0.038 | 3.1 |
| 9 | 0.8991 | 0.4948 | 1.7725 | 0.5416 | 0.340 | 0.296 | 0.046 | 0.040 | 3.1 |
| 10 | 0.8789 | 0.4727 | 1.8813 | 0.5229 | 0.349 | 0.296 | 0.045 | 0.037 | 3.1 |

## Last finished epoch

Epoch 10.

Segment scores

| class | precision | recall | f1 | tp | fp | fn |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Alarm_bell_ringing | 0.099 | 0.455 | 0.162 | 81 | 740 | 97 |
| Blender | 0.261 | 0.838 | 0.398 | 191 | 542 | 37 |
| Cat | 0.086 | 0.788 | 0.156 | 93 | 984 | 25 |
| Dishes | 0.110 | 0.742 | 0.191 | 115 | 932 | 40 |
| Dog | 0.086 | 0.945 | 0.158 | 86 | 913 | 5 |
| Electric_shaver_toothbrush | 0.188 | 0.929 | 0.313 | 157 | 677 | 12 |
| Frying | 0.196 | 0.750 | 0.311 | 177 | 725 | 59 |
| Running_water | 0.116 | 0.844 | 0.204 | 119 | 908 | 22 |
| Speech | 0.739 | 0.901 | 0.812 | 985 | 348 | 108 |
| Vacuum_cleaner | 0.151 | 0.753 | 0.251 | 143 | 805 | 47 |
| micro | 0.221 | 0.826 | 0.349 | 2147 | 7574 | 452 |
| macro f1 |  |  | 0.296 |  |  |  |

Event scores

| class | precision | recall | f1 | tp | fp | fn |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Alarm_bell_ringing | 0.003 | 0.061 | 0.006 | 2 | 652 | 31 |
| Blender | 0.008 | 0.160 | 0.016 | 4 | 468 | 21 |
| Cat | 0.004 | 0.079 | 0.007 | 3 | 790 | 35 |
| Dishes | 0.044 | 0.239 | 0.074 | 26 | 568 | 83 |
| Dog | 0.024 | 0.432 | 0.046 | 16 | 639 | 21 |
| Electric_shaver_toothbrush | 0.010 | 0.217 | 0.019 | 5 | 504 | 18 |
| Frying | 0.012 | 0.222 | 0.023 | 6 | 491 | 21 |
| Running_water | 0.008 | 0.250 | 0.016 | 5 | 600 | 15 |
| Speech | 0.109 | 0.249 | 0.152 | 78 | 636 | 235 |
| Vacuum_cleaner | 0.008 | 0.211 | 0.015 | 4 | 524 | 15 |
| micro | 0.025 | 0.231 | 0.045 | 149 | 5872 | 495 |
| macro f1 |  |  | 0.037 |  |  |  |
