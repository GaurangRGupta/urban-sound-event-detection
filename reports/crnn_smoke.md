# Small convolutional recurrent network

This file is a pipeline check on a prefix of the synthetic clips. It is not the full-data CRNN score.

The convolutional front end treats the log-mel matrix as a time-frequency grid. That is the image-style step this project had set aside. The recurrent layer is the same kind of frame GRU used in the previous model. Compare the scores with that GRU before reading this network as an improvement.

Command: `uv run python -m nndl_project.train_crnn --limit 200`

Started: 2026-10-08T02:04:12+05:30
Updated: 2026-10-08T02:08:14+05:30
Interrupted: False
Device: cpu
Project root: `C:\Users\gaura\Documents\grok_build_trial\neural_networks_and_deep_learning_project`

## Setup

- Train clips: 200, 0.wav through 199.wav.
- Validation clips: 200, 0.wav through 199.wav.
- Network: conv 16 then 32, kernel 3 by 3, ReLU, max-pool frequency by 4 twice, then bidirectional GRU, hidden 64 in each direction, 1 layer, then linear 128 to 10 logits. Time stays at 497 frames. Sigmoid per class, inside the loss.
- Parameters: 80586.
- Loss: binary cross-entropy on logits. The printed cost is this weighted loss. Positive weight for a class is negative frames divided by positive frames: Alarm_bell_ringing 82.95, Blender 33.57, Cat 26.23, Dishes 26.88, Dog 37.29, Electric_shaver_toothbrush 13.14, Frying 6.23, Running_water 12.27, Speech 3.02, Vacuum_cleaner 7.05.
- Optimizer: Adam, learning rate 0.001, weight decay 0.0, batch 16 clips, seed 0.
- Gradient clip: global norm 5.0, applied after backward and before the Adam step. The logged grad_norm is the norm before that clip.
- An iteration is one training batch. This run has stored 130 iteration costs.
- Cost figure: `C:\Users\gaura\Documents\grok_build_trial\neural_networks_and_deep_learning_project\reports\crnn_smoke_cost.png`. The top panel is the training cost of each iteration. The bottom panel is the mean training cost and the validation cost at the end of each epoch. The same figure is at `C:\Users\gaura\Documents\grok_build_trial\neural_networks_and_deep_learning_project\data\cache\crnn\limit200_val200\cost.png`.
- Decode: median filter of 11 frames, threshold 0.5. The threshold was not chosen on validation.
- Designated model: the last finished epoch. Validation F1 is not used to pick a checkpoint.
- Feature cache version 1: 64 mel bands, FFT 1024, hop 320 at 16000 Hz, 497 frames.
- Standardization: training-set mean and standard deviation per mel band.
- PyTorch 2.14.1+cpu, NumPy 2.5.3.
- Checkpoint: `C:\Users\gaura\Documents\grok_build_trial\neural_networks_and_deep_learning_project\data\cache\crnn\limit200_val200\model_last.pt`.
- Log: `C:\Users\gaura\Documents\grok_build_trial\neural_networks_and_deep_learning_project\data\cache\crnn\limit200_val200\train.log`.
- Epochs requested: 10. Epochs finished: 10.

Logistic baseline on all 2500 validation clips: segment micro F1 0.393, segment macro F1 0.289, event micro F1 0.044, event macro F1 0.03.

Context MLP epoch 432 of 1000, interrupted, on all 2500 validation clips: segment micro F1 0.398, segment macro F1 0.349, event micro F1 0.053, event macro F1 0.041.

Frame GRU epoch 30 of 30 on all 2500 validation clips: segment micro F1 0.509, segment macro F1 0.438, event micro F1 0.104, event macro F1 0.096.

Compare this file with those three lines only when this run also scores all 2500 validation clips and uses the planned settings (16 channels, hidden 64, learning rate 0.001, weight decay 0, gradient clip 5, batch 16, seed 0).

## First validation clip

Clip: 0.wav

Reference labels:

- Dog from 5.050 s to 5.310 s
- Speech from 6.832 s to 8.674 s
- Dog from 7.412 s to 7.797 s

Model events from the last finished epoch:

- Alarm_bell_ringing from 2.920 s to 3.140 s, score 0.571
- Alarm_bell_ringing from 3.420 s to 4.280 s, score 0.568
- Alarm_bell_ringing from 4.520 s to 5.780 s, score 0.626
- Alarm_bell_ringing from 5.980 s to 9.320 s, score 0.593
- Blender from 7.420 s to 7.540 s, score 0.500
- Blender from 7.800 s to 8.080 s, score 0.517
- Blender from 8.340 s to 8.480 s, score 0.503
- Cat from 0.000 s to 0.020 s, score 0.517
- Cat from 2.800 s to 9.940 s, score 0.680
- Dishes from 8.200 s to 8.480 s, score 0.550
- Dog from 2.900 s to 9.940 s, score 0.703
- Speech from 2.920 s to 3.200 s, score 0.618
- Speech from 3.500 s to 4.340 s, score 0.598
- Speech from 4.580 s to 5.860 s, score 0.639
- Speech from 6.020 s to 9.940 s, score 0.670
- Vacuum_cleaner from 4.580 s to 4.740 s, score 0.519
- Vacuum_cleaner from 4.940 s to 5.220 s, score 0.532
- Vacuum_cleaner from 6.480 s to 6.740 s, score 0.516
- Vacuum_cleaner from 6.920 s to 7.700 s, score 0.537
- Vacuum_cleaner from 7.860 s to 8.160 s, score 0.535
- Vacuum_cleaner from 8.360 s to 8.500 s, score 0.503
- Vacuum_cleaner from 8.780 s to 8.900 s, score 0.508
- Vacuum_cleaner from 9.520 s to 9.660 s, score 0.503

## Epoch history

| epoch | train weighted loss | train unweighted loss | val weighted loss | val unweighted loss | segment micro f1 | segment macro f1 | event micro f1 | event macro f1 | seconds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.2487 | 0.6459 | 1.7415 | 0.6462 | 0.272 | 0.243 | 0.036 | 0.034 | 22.3 |
| 2 | 1.1635 | 0.6525 | 1.6786 | 0.6856 | 0.272 | 0.245 | 0.041 | 0.038 | 22.6 |
| 3 | 1.1298 | 0.6495 | 1.6304 | 0.6534 | 0.289 | 0.255 | 0.044 | 0.042 | 22.2 |
| 4 | 1.0831 | 0.6021 | 1.6137 | 0.6149 | 0.312 | 0.271 | 0.052 | 0.048 | 23.7 |
| 5 | 1.0510 | 0.6202 | 1.6017 | 0.6156 | 0.320 | 0.278 | 0.059 | 0.057 | 24.8 |
| 6 | 1.0153 | 0.5683 | 1.4791 | 0.6388 | 0.333 | 0.286 | 0.061 | 0.063 | 18.7 |
| 7 | 0.9894 | 0.5789 | 1.5790 | 0.5861 | 0.351 | 0.296 | 0.073 | 0.068 | 24.1 |
| 8 | 0.9579 | 0.5770 | 1.4867 | 0.5881 | 0.356 | 0.309 | 0.071 | 0.070 | 24.6 |
| 9 | 0.9313 | 0.5349 | 1.4756 | 0.5816 | 0.368 | 0.313 | 0.073 | 0.073 | 24.2 |
| 10 | 0.9017 | 0.5327 | 1.4959 | 0.5929 | 0.367 | 0.315 | 0.075 | 0.076 | 23.6 |

## Last finished epoch

Epoch 10.

Segment scores

| class | precision | recall | f1 | tp | fp | fn |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Alarm_bell_ringing | 0.164 | 0.764 | 0.271 | 136 | 691 | 42 |
| Blender | 0.182 | 0.947 | 0.305 | 216 | 972 | 12 |
| Cat | 0.105 | 0.839 | 0.186 | 99 | 847 | 19 |
| Dishes | 0.101 | 0.774 | 0.178 | 120 | 1074 | 35 |
| Dog | 0.098 | 0.945 | 0.178 | 86 | 789 | 5 |
| Electric_shaver_toothbrush | 0.191 | 0.852 | 0.312 | 144 | 610 | 25 |
| Frying | 0.287 | 0.860 | 0.431 | 203 | 504 | 33 |
| Running_water | 0.116 | 0.894 | 0.206 | 126 | 957 | 15 |
| Speech | 0.710 | 0.946 | 0.811 | 1034 | 422 | 59 |
| Vacuum_cleaner | 0.159 | 0.968 | 0.273 | 184 | 975 | 6 |
| micro | 0.230 | 0.903 | 0.367 | 2348 | 7841 | 251 |
| macro f1 |  |  | 0.315 |  |  |  |

Event scores

| class | precision | recall | f1 | tp | fp | fn |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Alarm_bell_ringing | 0.014 | 0.152 | 0.025 | 5 | 361 | 28 |
| Blender | 0.036 | 0.560 | 0.068 | 14 | 372 | 11 |
| Cat | 0.019 | 0.158 | 0.033 | 6 | 317 | 32 |
| Dishes | 0.040 | 0.156 | 0.064 | 17 | 404 | 92 |
| Dog | 0.025 | 0.216 | 0.046 | 8 | 306 | 29 |
| Electric_shaver_toothbrush | 0.071 | 0.435 | 0.122 | 10 | 131 | 13 |
| Frying | 0.076 | 0.407 | 0.128 | 11 | 134 | 16 |
| Running_water | 0.047 | 0.450 | 0.085 | 9 | 184 | 11 |
| Speech | 0.111 | 0.153 | 0.129 | 48 | 385 | 265 |
| Vacuum_cleaner | 0.034 | 0.526 | 0.064 | 10 | 284 | 9 |
| micro | 0.046 | 0.214 | 0.075 | 138 | 2878 | 506 |
| macro f1 |  |  | 0.076 |  |  |  |
