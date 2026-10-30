# Small convolutional recurrent network

These scores are the small convolutional recurrent network on the clips named below. The model is the last finished epoch. The threshold was not tuned.

The convolutional front end treats the log-mel matrix as a time-frequency grid. That is the image-style step this project had set aside. The recurrent layer is the same kind of frame GRU used in the previous model. Compare the scores with that GRU before reading this network as an improvement.

Command: `uv run python -m nndl_project.train_crnn`

Started: 2026-10-08T02:10:31+05:30
Updated: 2026-10-08T04:50:15+05:30
Interrupted: False
Device: cpu
Project root: `C:\Users\gaura\Documents\grok_build_trial\neural_networks_and_deep_learning_project`

## Setup

- Train clips: 10000, 0.wav through 9999.wav.
- Validation clips: 2500, 0.wav through 2499.wav.
- Network: conv 16 then 32, kernel 3 by 3, ReLU, max-pool frequency by 4 twice, then bidirectional GRU, hidden 64 in each direction, 1 layer, then linear 128 to 10 logits. Time stays at 497 frames. Sigmoid per class, inside the loss.
- Parameters: 80586.
- Loss: binary cross-entropy on logits. The printed cost is this weighted loss. Positive weight for a class is negative frames divided by positive frames: Alarm_bell_ringing 32.26, Blender 22.26, Cat 37.00, Dishes 30.58, Dog 35.62, Electric_shaver_toothbrush 9.50, Frying 6.89, Running_water 11.20, Speech 2.91, Vacuum_cleaner 9.10.
- Optimizer: Adam, learning rate 0.001, weight decay 0.0, batch 16 clips, seed 0.
- Gradient clip: global norm 5.0, applied after backward and before the Adam step. The logged grad_norm is the norm before that clip.
- An iteration is one training batch. This run has stored 6250 iteration costs.
- Cost figure: `C:\Users\gaura\Documents\grok_build_trial\neural_networks_and_deep_learning_project\reports\crnn_cost.png`. The top panel is the training cost of each iteration. The bottom panel is the mean training cost and the validation cost at the end of each epoch. The same figure is at `C:\Users\gaura\Documents\grok_build_trial\neural_networks_and_deep_learning_project\data\cache\crnn\full\cost.png`.
- Decode: median filter of 11 frames, threshold 0.5. The threshold was not chosen on validation.
- Designated model: the last finished epoch. Validation F1 is not used to pick a checkpoint.
- Feature cache version 1: 64 mel bands, FFT 1024, hop 320 at 16000 Hz, 497 frames.
- Standardization: training-set mean and standard deviation per mel band.
- PyTorch 2.14.1+cpu, NumPy 2.5.3.
- Checkpoint: `C:\Users\gaura\Documents\grok_build_trial\neural_networks_and_deep_learning_project\data\cache\crnn\full\model_last.pt`.
- Log: `C:\Users\gaura\Documents\grok_build_trial\neural_networks_and_deep_learning_project\data\cache\crnn\full\train.log`.
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

- Dog from 4.980 s to 5.520 s, score 0.687
- Dog from 6.960 s to 7.900 s, score 0.776
- Dog from 9.480 s to 9.820 s, score 0.663
- Speech from 5.060 s to 5.220 s, score 0.547
- Speech from 6.260 s to 8.700 s, score 0.822

## Epoch history

| epoch | train weighted loss | train unweighted loss | val weighted loss | val unweighted loss | segment micro f1 | segment macro f1 | event micro f1 | event macro f1 | seconds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.9076 | 0.5156 | 1.0124 | 0.4748 | 0.449 | 0.402 | 0.119 | 0.117 | 891.4 |
| 2 | 0.7010 | 0.3958 | 0.8768 | 0.4025 | 0.499 | 0.441 | 0.126 | 0.125 | 883.1 |
| 3 | 0.6097 | 0.3434 | 0.8587 | 0.4101 | 0.503 | 0.464 | 0.121 | 0.135 | 960.2 |
| 4 | 0.5472 | 0.3067 | 0.8148 | 0.4147 | 0.506 | 0.450 | 0.121 | 0.133 | 962.8 |
| 5 | 0.5036 | 0.2819 | 0.9144 | 0.3245 | 0.562 | 0.498 | 0.152 | 0.159 | 966.3 |
| 6 | 0.4675 | 0.2607 | 0.9279 | 0.3326 | 0.556 | 0.495 | 0.141 | 0.149 | 997.9 |
| 7 | 0.4403 | 0.2450 | 0.9592 | 0.3424 | 0.556 | 0.505 | 0.141 | 0.156 | 994.2 |
| 8 | 0.4117 | 0.2296 | 0.9766 | 0.3077 | 0.588 | 0.523 | 0.159 | 0.160 | 979.1 |
| 9 | 0.3924 | 0.2178 | 0.8995 | 0.2933 | 0.600 | 0.546 | 0.149 | 0.163 | 970.4 |
| 10 | 0.3716 | 0.2068 | 1.0884 | 0.2913 | 0.612 | 0.542 | 0.142 | 0.155 | 965.7 |

## Last finished epoch

Epoch 10.

Segment scores

| class | precision | recall | f1 | tp | fp | fn |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Alarm_bell_ringing | 0.480 | 0.831 | 0.608 | 1860 | 2016 | 379 |
| Blender | 0.420 | 0.550 | 0.476 | 1322 | 1826 | 1083 |
| Cat | 0.227 | 0.899 | 0.363 | 1222 | 4157 | 137 |
| Dishes | 0.218 | 0.808 | 0.343 | 1509 | 5427 | 359 |
| Dog | 0.213 | 0.812 | 0.337 | 1156 | 4281 | 268 |
| Electric_shaver_toothbrush | 0.406 | 0.898 | 0.559 | 1885 | 2760 | 214 |
| Frying | 0.658 | 0.825 | 0.732 | 2894 | 1504 | 615 |
| Running_water | 0.467 | 0.761 | 0.578 | 1861 | 2127 | 586 |
| Speech | 0.770 | 0.914 | 0.836 | 12158 | 3636 | 1147 |
| Vacuum_cleaner | 0.427 | 0.943 | 0.588 | 2368 | 3181 | 142 |
| micro | 0.477 | 0.851 | 0.612 | 28235 | 30915 | 4930 |
| macro f1 |  |  | 0.542 |  |  |  |

Event scores

| class | precision | recall | f1 | tp | fp | fn |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Alarm_bell_ringing | 0.074 | 0.306 | 0.119 | 132 | 1660 | 299 |
| Blender | 0.078 | 0.350 | 0.128 | 93 | 1099 | 173 |
| Cat | 0.027 | 0.140 | 0.045 | 60 | 2153 | 369 |
| Dishes | 0.045 | 0.141 | 0.069 | 184 | 3867 | 1125 |
| Dog | 0.031 | 0.147 | 0.051 | 81 | 2539 | 469 |
| Electric_shaver_toothbrush | 0.123 | 0.507 | 0.198 | 145 | 1034 | 141 |
| Frying | 0.246 | 0.660 | 0.358 | 249 | 764 | 128 |
| Running_water | 0.094 | 0.428 | 0.154 | 131 | 1265 | 175 |
| Speech | 0.149 | 0.244 | 0.185 | 959 | 5482 | 2968 |
| Vacuum_cleaner | 0.146 | 0.745 | 0.244 | 187 | 1093 | 64 |
| micro | 0.096 | 0.273 | 0.142 | 2221 | 20956 | 5911 |
| macro f1 |  |  | 0.155 |  |  |  |
