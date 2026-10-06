# Frame gated recurrent unit

These scores are the frame gated recurrent unit on the clips named below. The model is the last finished epoch. The threshold was not tuned.

Command: `uv run python -m nndl_project.train_frame_gru`

Started: 2026-10-05T15:00:37+05:30
Updated: 2026-10-05T21:13:58+05:30
Interrupted: False
Device: cpu
Project root: `C:\Users\gaura\Documents\grok_build_trial\neural_networks_and_deep_learning_project`

## Setup

- Train clips: 10000, 0.wav through 9999.wav.
- Validation clips: 2500, 0.wav through 2499.wav.
- Network: bidirectional GRU, input 64, hidden 64 in each direction, 1 layer, then linear 128 to 10 logits. Sigmoid per class, inside the loss.
- Parameters: 51210.
- Loss: binary cross-entropy on logits. Positive weight for a class is negative frames divided by positive frames: Alarm_bell_ringing 32.26, Blender 22.26, Cat 37.00, Dishes 30.58, Dog 35.62, Electric_shaver_toothbrush 9.50, Frying 6.89, Running_water 11.20, Speech 2.91, Vacuum_cleaner 9.10.
- Optimizer: Adam, learning rate 0.001, weight decay 0.0, batch 16 clips, seed 0.
- Gradient clip: global norm 5.0, applied after backward and before the Adam step. The logged grad_norm is the norm before that clip.
- Decode: median filter of 11 frames, threshold 0.5. The threshold was not chosen on validation.
- Designated model: the last finished epoch. Validation F1 is not used to pick a checkpoint.
- Feature cache version 1: 64 mel bands, FFT 1024, hop 320 at 16000 Hz, 497 frames.
- Standardization: training-set mean and standard deviation per mel band.
- PyTorch 2.14.1+cpu, NumPy 2.5.3.
- Checkpoint: `C:\Users\gaura\Documents\grok_build_trial\neural_networks_and_deep_learning_project\data\cache\frame_gru\full\model_last.pt`.
- Log: `C:\Users\gaura\Documents\grok_build_trial\neural_networks_and_deep_learning_project\data\cache\frame_gru\full\train.log`.
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

- Alarm_bell_ringing from 2.900 s to 3.020 s, score 0.536
- Alarm_bell_ringing from 8.220 s to 8.400 s, score 0.519
- Dog from 4.980 s to 5.460 s, score 0.548
- Dog from 9.460 s to 9.840 s, score 0.722
- Speech from 4.660 s to 5.320 s, score 0.539
- Speech from 6.880 s to 8.660 s, score 0.791

## Epoch history

| epoch | train weighted loss | train unweighted loss | val weighted loss | val unweighted loss | segment micro f1 | segment macro f1 | event micro f1 | event macro f1 | seconds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.9733 | 0.5493 | 1.2770 | 0.5306 | 0.382 | 0.335 | 0.076 | 0.077 | 654.6 |
| 2 | 0.8408 | 0.4711 | 1.2167 | 0.5071 | 0.402 | 0.354 | 0.089 | 0.089 | 688.8 |
| 3 | 0.7799 | 0.4386 | 1.2634 | 0.4768 | 0.411 | 0.365 | 0.092 | 0.094 | 709.1 |
| 4 | 0.7309 | 0.4100 | 1.2747 | 0.4871 | 0.405 | 0.357 | 0.085 | 0.087 | 750.3 |
| 5 | 0.6949 | 0.3894 | 1.3116 | 0.4758 | 0.414 | 0.366 | 0.086 | 0.087 | 736.3 |
| 6 | 0.6636 | 0.3718 | 1.3390 | 0.4635 | 0.429 | 0.376 | 0.090 | 0.089 | 731.2 |
| 7 | 0.6345 | 0.3559 | 1.3593 | 0.4582 | 0.434 | 0.385 | 0.093 | 0.091 | 713.6 |
| 8 | 0.6113 | 0.3430 | 1.4314 | 0.4178 | 0.447 | 0.397 | 0.103 | 0.100 | 711.4 |
| 9 | 0.5891 | 0.3295 | 1.3731 | 0.4231 | 0.452 | 0.397 | 0.095 | 0.093 | 716.3 |
| 10 | 0.5675 | 0.3178 | 1.4239 | 0.4298 | 0.455 | 0.396 | 0.093 | 0.093 | 715.7 |
| 11 | 0.5494 | 0.3084 | 1.4011 | 0.4094 | 0.466 | 0.413 | 0.096 | 0.093 | 711.3 |
| 12 | 0.5351 | 0.2994 | 1.4327 | 0.4041 | 0.470 | 0.408 | 0.093 | 0.091 | 718.5 |
| 13 | 0.5216 | 0.2916 | 1.5099 | 0.4186 | 0.459 | 0.399 | 0.094 | 0.089 | 717.1 |
| 14 | 0.5059 | 0.2837 | 1.5458 | 0.4156 | 0.465 | 0.404 | 0.095 | 0.089 | 718.3 |
| 15 | 0.4933 | 0.2758 | 1.4220 | 0.4043 | 0.470 | 0.416 | 0.099 | 0.096 | 714.6 |
| 16 | 0.4790 | 0.2690 | 1.7680 | 0.4054 | 0.484 | 0.405 | 0.099 | 0.092 | 713.0 |
| 17 | 0.4723 | 0.2644 | 1.4769 | 0.3965 | 0.482 | 0.422 | 0.099 | 0.094 | 714.4 |
| 18 | 0.4580 | 0.2567 | 1.4504 | 0.4056 | 0.472 | 0.422 | 0.100 | 0.097 | 718.6 |
| 19 | 0.4503 | 0.2525 | 1.5549 | 0.4027 | 0.486 | 0.423 | 0.096 | 0.095 | 720.7 |
| 20 | 0.4433 | 0.2494 | 1.6521 | 0.4091 | 0.475 | 0.417 | 0.098 | 0.093 | 716.2 |
| 21 | 0.4364 | 0.2449 | 1.7062 | 0.4014 | 0.477 | 0.420 | 0.097 | 0.092 | 715.1 |
| 22 | 0.4216 | 0.2370 | 1.7957 | 0.3865 | 0.489 | 0.420 | 0.097 | 0.090 | 916.2 |
| 23 | 0.4214 | 0.2362 | 1.6827 | 0.4021 | 0.485 | 0.422 | 0.102 | 0.095 | 862.6 |
| 24 | 0.4121 | 0.2317 | 1.6888 | 0.4005 | 0.491 | 0.428 | 0.105 | 0.095 | 886.0 |
| 25 | 0.4020 | 0.2266 | 1.6861 | 0.4033 | 0.487 | 0.421 | 0.099 | 0.094 | 864.2 |
| 26 | 0.4018 | 0.2261 | 1.6679 | 0.3963 | 0.492 | 0.427 | 0.105 | 0.096 | 855.4 |
| 27 | 0.3885 | 0.2198 | 1.8758 | 0.3815 | 0.506 | 0.431 | 0.102 | 0.095 | 843.6 |
| 28 | 0.3888 | 0.2189 | 1.7540 | 0.3937 | 0.494 | 0.430 | 0.104 | 0.095 | 762.3 |
| 29 | 0.3825 | 0.2166 | 1.9162 | 0.3957 | 0.495 | 0.427 | 0.101 | 0.094 | 704.7 |
| 30 | 0.3732 | 0.2108 | 1.8788 | 0.3768 | 0.509 | 0.438 | 0.104 | 0.096 | 698.2 |

## Last finished epoch

Epoch 30.

Segment scores

| class | precision | recall | f1 | tp | fp | fn |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Alarm_bell_ringing | 0.195 | 0.453 | 0.272 | 1015 | 4198 | 1224 |
| Blender | 0.285 | 0.438 | 0.345 | 1054 | 2646 | 1351 |
| Cat | 0.202 | 0.815 | 0.323 | 1108 | 4384 | 251 |
| Dishes | 0.166 | 0.702 | 0.269 | 1311 | 6584 | 557 |
| Dog | 0.171 | 0.882 | 0.287 | 1256 | 6073 | 168 |
| Electric_shaver_toothbrush | 0.372 | 0.899 | 0.526 | 1886 | 3184 | 213 |
| Frying | 0.531 | 0.718 | 0.611 | 2519 | 2223 | 990 |
| Running_water | 0.278 | 0.638 | 0.387 | 1560 | 4052 | 887 |
| Speech | 0.790 | 0.819 | 0.804 | 10893 | 2889 | 2412 |
| Vacuum_cleaner | 0.429 | 0.806 | 0.560 | 2022 | 2694 | 488 |
| micro | 0.387 | 0.742 | 0.509 | 24624 | 38927 | 8541 |
| macro f1 |  |  | 0.438 |  |  |  |

Event scores

| class | precision | recall | f1 | tp | fp | fn |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Alarm_bell_ringing | 0.008 | 0.060 | 0.014 | 26 | 3175 | 405 |
| Blender | 0.022 | 0.169 | 0.040 | 45 | 1959 | 221 |
| Cat | 0.024 | 0.156 | 0.042 | 67 | 2722 | 362 |
| Dishes | 0.030 | 0.109 | 0.047 | 143 | 4633 | 1166 |
| Dog | 0.014 | 0.093 | 0.025 | 51 | 3511 | 499 |
| Electric_shaver_toothbrush | 0.087 | 0.444 | 0.146 | 127 | 1326 | 159 |
| Frying | 0.123 | 0.459 | 0.194 | 173 | 1237 | 204 |
| Running_water | 0.034 | 0.265 | 0.060 | 81 | 2306 | 225 |
| Speech | 0.161 | 0.282 | 0.205 | 1108 | 5778 | 2819 |
| Vacuum_cleaner | 0.110 | 0.606 | 0.186 | 152 | 1232 | 99 |
| micro | 0.066 | 0.243 | 0.104 | 1973 | 27879 | 6159 |
| macro f1 |  |  | 0.096 |  |  |  |
