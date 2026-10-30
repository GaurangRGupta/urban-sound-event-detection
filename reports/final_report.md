# Urban sound event detection with timestamps: final report

Deeptha Dinakar (J010) and Gaurang Gupta (J013). Neural Networks and Deep Learning.

## 1. Problem

The input is a recording of about 10 seconds, one channel. The system says which everyday sounds are present, and when each one starts and ends. Several sounds can overlap. The output for a clip is a list of events, each with a class name, a start time in seconds, an end time in seconds, and a score between 0 and 1. This is timestamped (strong) sound event detection, not clip tagging.

The ten output classes are fixed by the problem statement: speech_crowd, car_horn, siren, engine_traffic, two_wheeler, construction, dog, music, bird, other.

## 2. Data

Only DESED (Domestic Environment Sound Event Detection, DCASE Task 4) has timestamps and audio we can train on. Its ten home classes are Alarm_bell_ringing, Blender, Cat, Dishes, Dog, Electric_shaver_toothbrush, Frying, Running_water, Speech, Vacuum_cleaner.

| Set | Clips | Labels | Role |
| --- | ---: | --- | --- |
| DESED synthetic21_train | 10,000 | strong (Scaper mixer wrote the times) | training |
| DESED synthetic21_validation | 2,500 | strong | validation; the only place anything is tuned |
| DESED public evaluation | 692 real clips | strong | test, scored once at the end |
| DCASE 2019 synthetic evaluation, 16 conditions | 750 to 783 each | strong | robustness test, scored once at the end |
| FSD50K | 51,197 | clip tags only, no audio downloaded | vocabulary check for street classes; not trained on |

Official splits are kept as published. Train and validation both contain a `0.wav`; each label is joined only to the wav in its own folder, and `tests/test_splits.py` checks that the test sets cannot be reached through the training loader.

### Label map

The models are trained on the ten DESED names and the street map is applied to their output (`src/nndl_project/label_map.py`, checked by `tests/test_label_map.py`):

| DESED class | Street class |
| --- | --- |
| Speech | speech_crowd |
| Dog | dog |
| Alarm_bell_ringing, Blender, Cat, Dishes, Electric_shaver_toothbrush, Frying, Running_water, Vacuum_cleaner | other |

DESED is a home dataset and the class list is street-leaning. A blender is not traffic, so nothing is forced onto a street class it does not belong to. car_horn, siren, engine_traffic, two_wheeler, construction, music and bird get no DESED label and therefore have no timestamp data in this project. FSD50K confirms those sounds exist in public data (horn, siren, engine, motorcycle, tools, music, bird), but only as clip tags, and we did not invent timestamps for them.

## 3. Method

1. **Pre-processing.** Each clip is mixed to one channel, resampled from 44.1 kHz to 16 kHz, and padded or cut to 10 s.
2. **Features.** Log-mel spectrogram: FFT 1024, hop 320 samples (20 ms), 64 mel bands up to 8 kHz. One clip is a 497 x 64 matrix. Each band is standardized with the training-set mean and standard deviation.
3. **Targets.** Each of the 497 frames gets a yes/no for each of the ten classes. A frame is yes when its analysis window overlaps a labeled event. Several classes can be yes at once (multi-label).
4. **Models.** Four models, each a stronger step than the last, with everything else held fixed (features, targets, class weights, decode, data):
   - **Logistic baseline**: ten independent logistic regressions on one frame at a time. No memory of neighboring frames.
   - **Context MLP**: one shared network on an 11-frame window (704 inputs, 128 hidden ReLU units, 10 sigmoid outputs).
   - **Frame GRU**: one bidirectional GRU, 64 units each way, over the whole 497-frame sequence, then a linear layer to 10 outputs. 51,210 parameters.
   - **Small CRNN**: two 3x3 convolutions (16 and 32 channels, ReLU, frequency max-pool by 4 each), then the same bidirectional GRU, then a linear layer. Time is never pooled. 80,586 parameters.
5. **Loss.** Binary cross-entropy per class per frame, with a positive weight of (negative frames / positive frames) per class so rare classes are not ignored. Adam, learning rate 0.001, batch of 16 clips, gradient norm clipped at 5. PyTorch on CPU.
6. **Decode.** Median filter along time (11 frames, about 220 ms), then a threshold of 0.5. Each run of active frames becomes one event, with the mean probability inside the run as its score.
7. **Metrics.** Segment F1 on 1-second bins, and event F1, where a predicted event matches a reference of the same class when its onset is within 200 ms and its offset is within max(200 ms, 20% of the event's length). Micro (pooled counts) and macro (mean of per-class F1) are both reported.
8. **Final evaluation.** The decode (one threshold per class, one median length) is tuned on validation only. It is then frozen, and the public test and the 16 DCASE 2019 folders are scored once (`src/nndl_project/evaluate.py`).

The problem statement planned a non-convolutional first model. The context MLP and the frame GRU are that model; the CRNN was added afterwards and is reported as an extra step.

## 4. Results on validation (2,500 clips, fixed decode)

| Model | Segment micro F1 | Segment macro F1 | Event micro F1 | Event macro F1 |
| --- | ---: | ---: | ---: | ---: |
| Logistic baseline | 0.393 | 0.289 | 0.044 | 0.030 |
| Context MLP (epoch 432) | 0.398 | 0.349 | 0.053 | 0.041 |
| Frame GRU (epoch 30) | 0.509 | 0.438 | 0.104 | 0.096 |
| Small CRNN (epoch 10) | **0.612** | **0.542** | **0.142** | **0.155** |

Event F1 per class:

| Class | Baseline | Context MLP | Frame GRU | CRNN |
| --- | ---: | ---: | ---: | ---: |
| Alarm_bell_ringing | 0.000 | 0.017 | 0.014 | 0.119 |
| Blender | 0.000 | 0.009 | 0.040 | 0.128 |
| Cat | 0.009 | 0.019 | 0.042 | 0.045 |
| Dishes | 0.049 | 0.061 | 0.047 | 0.069 |
| Dog | 0.044 | 0.035 | 0.025 | 0.051 |
| Electric_shaver_toothbrush | 0.018 | 0.017 | 0.146 | 0.198 |
| Frying | 0.029 | 0.024 | 0.194 | 0.358 |
| Running_water | 0.004 | 0.006 | 0.060 | 0.154 |
| Speech | 0.146 | 0.212 | 0.205 | 0.185 |
| Vacuum_cleaner | 0.004 | 0.006 | 0.186 | 0.244 |

What the numbers say:

- **Context matters more than capacity.** The MLP sees 220 ms and barely moves event F1. The GRU sees the whole clip and doubles it. Event F1 is about edges, and a model that knows what came before and after can place them.
- **The convolution helps further.** The CRNN improves every score again. It reaches the best event F1 for nine of the ten classes; the context MLP is slightly better on Speech.
- **Long, steady sounds are easiest.** Frying, Vacuum_cleaner and Electric_shaver_toothbrush score highest on events. Short, sharp sounds (Cat, Dog, Dishes) stay below 0.07: a bark of a few hundred milliseconds has to hit a 200 ms onset collar.
- **Precision is the weak side.** The CRNN's event recall is 0.27 but precision is 0.10. Most errors are extra short events and split events, which is what the median filter and the threshold control. That is why tuning the decode on validation is the next step.
- **The networks overfit.** For the GRU and CRNN, the validation weighted loss rises after a few epochs while training loss keeps falling (`reports/crnn_cost.png`). No dropout, augmentation or early stopping was used, and the designated model is always the last epoch.

Example, validation clip `0.wav` (reference: Dog 5.05 to 5.31 s, Speech 6.83 to 8.67 s, Dog 7.41 to 7.80 s). The baseline emits 22 events, including six spurious Cat events. The CRNN emits five, and they sit on the right sounds: Speech 6.26 to 8.70 s, Dog 4.98 to 5.52 s, Dog 6.96 to 7.90 s, plus a short Speech blip and a Dog at 9.5 s. Yet none of them counts as an event match. The Speech onset is 0.57 s early, and the first Dog ends 0.21 s late against a 0.20 s offset collar. This clip shows why segment F1 (0.61) is so much higher than event F1 (0.14): the model finds the sounds, but the edges have to be within 200 ms.

## 5. Held-out results

Produced by:

```powershell
uv run python -m nndl_project.evaluate
```

The script writes `reports/final_evaluation_crnn.md` with: the tuned thresholds; validation, public test and each DCASE 2019 condition, each with fixed and tuned decode; scores in DESED names and in street names; and per-class tables for the public test. Copy its headline table here once it has run.

What to look for when reading it:

- The **public test** is real YouTube audio; the model has only heard synthetic mixes. A drop from validation is the synthetic-to-real gap, and it is expected.
- **fbsnr_30dB → 0dB** shows how scores fall as the event gets quieter against the background.
- **500ms / 5500ms / 9500ms** show whether events near the clip edges are found.
- The **distorted_*** folders show robustness to clipping, compression, filtering and phone playback.

## 6. Success criteria

| Criterion (problem statement) | Status |
| --- | --- |
| 1. The label map is written down and checked by a test. Official splits unchanged. | Done: `label_map.py`, `tests/test_label_map.py`; `tests/test_splits.py` guards the splits. |
| 2. For a clip, the system returns a list of events with label, start, end and score. | Done: `python -m nndl_project.predict clip.wav [--street] [--json out.json]`. |
| 3. On classes with timestamp labels, the first model has higher event F1 than the simple baseline, reported for every class. | Met on validation: the GRU beats the baseline on event micro/macro F1, and the CRNN beats it on all ten classes (table above). |
| 4. No numeric event-F1 target for car_horn, siren, engine_traffic, two_wheeler, construction, music, bird. | Respected: these are listed as having no timestamp labels and are not scored. |

## 7. Limitations

- The training audio is synthetic and from homes. Street sounds other than speech and dogs are not learned, only mapped to `other` when they are DESED classes.
- The clip-level (weak-label) loss from the method was not trained: the FSD50K and DESED weak-label waves were not downloaded.
- No regularization or early stopping; the GRU and CRNN overfit after a few epochs.
- CPU only, so models are small (51k to 81k parameters) and training was short (10 to 30 epochs).

Natural next steps: dropout and SpecAugment-style masking, mean-teacher training on the DESED unlabeled clips, and strongly labeled street data (for example URBAN-SED) for the seven missing street classes.

## 8. Reproduce

```powershell
uv sync
dvc pull -j 2
uv run pytest
uv run python -m nndl_project.baseline
uv run python -m nndl_project.train_context_mlp --epochs 432
uv run python -m nndl_project.train_frame_gru
uv run python -m nndl_project.train_crnn
uv run python -m nndl_project.evaluate
uv run python -m nndl_project.predict path\to\clip.wav --street --thresholds reports\final_evaluation_crnn.json
```
