# Architecture and design

This note describes the sound-event pipeline that is in the repository now. It covers the logistic baseline, the context multilayer perceptron (MLP) that has been trained, and the frame gated recurrent unit (GRU) that is next. The deep learning framework is PyTorch.

## 1. The problem

The task is multi-label classification on short time frames.

A clip is about 10 seconds of everyday sound. Several sounds can overlap. For every frame of 20 milliseconds, each of the ten Domestic Environment Sound Event Detection (DESED) classes is labeled yes or no. More than one class may be yes in the same frame. The model outputs a probability for each class.

The start time and end time of an event are not regression targets. The training loss never sees a numeric onset or offset as the thing it must predict. Those times are used only to paint the yes/no label on each frame. After classification, frames that stayed active are grouped into an interval. That interval is the event the scorer sees.

The ten class names in this pipeline are the DESED names:

`Alarm_bell_ringing`, `Blender`, `Cat`, `Dishes`, `Dog`, `Electric_shaver_toothbrush`, `Frying`, `Running_water`, `Speech`, `Vacuum_cleaner`.

The course street map is not applied here. Under that map, Speech would become speech_crowd, Dog would become dog, and the other eight DESED names would become other. The logistic baseline, the context MLP, and the frame GRU all keep the ten DESED names, so the scores can be compared class by class.

## 2. Inputs and outputs

One training or validation clip moves through these arrays.

| Stage | Shape and type | Meaning |
| --- | --- | --- |
| Wave on disk | 10 seconds, one channel, 44100 Hz | The synthetic file, for example `0.wav` |
| Wave after loading | `float32` vector of length 160000 | One channel at 16000 Hz |
| Log-mel matrix | `float32`, shape `(497, 64)` | 497 frames, 64 mel bands |
| Frame targets | integers 0 or 1, shape `(497, 10)` | One yes/no column per DESED class |
| Frame probabilities | `float32`, shape `(497, 10)` | One probability per class per frame |
| Decoded event | label string, start seconds, end seconds, score from 0 to 1 | One detected sound interval |

Log-mel means the logarithm of the mel-filterbank energy. The settings are a fast Fourier transform (FFT) length of 1024 samples, a hop of 320 samples (20 milliseconds), 64 bands, and a highest frequency of 8000 Hz. A frame is a row of 64 numbers. The matrix is the input to the classifier. It is a loudness view over time, and the model does not treat it as a picture.

A frame is positive for a class when its analysis window overlaps a strong label. The window for frame `i` covers samples `[i * 320, i * 320 + 1024)`. Overlap is used so a short event is not dropped between frame centers.

A strong label row has four fields: `filename`, `onset` in seconds, `offset` in seconds, and `event_label`.

## 3. Which data is used

Training reads only `synthetic21_train`: 10000 clips and the strong-label table `soundscapes.tsv` (32096 events). Scoring reads only `synthetic21_validation`: 2500 clips and its own `soundscapes.tsv` (8132 events). Both folders number files from `0.wav`. The same filename in the two folders is a different recording. Each label table is joined only to the wav files in its own folder.

The loader resamples 44100 Hz to 16000 Hz. There is no `soundscapes_16k` copy on disk.

These sets stay closed. They are not used to fit weights, to pick the threshold, or to report the baseline score. DCASE means Detection and Classification of Acoustic Scenes and Events. A `.tsv` file is a tab-separated table.

| Set | What it is | Why it stays closed |
| --- | --- | --- |
| `audio/eval/public` with `metadata/eval/public.tsv` | 692 real clips, 2765 events | This is the real public test. |
| `dcase2019/dataset`, 16 condition folders | Synthetic test under changes of noise and distortion. `fbsnr_30dB` is the clean reference inside that test | This is a robustness test. The same scene id is repeated across folders, so the folders are not extra training mass. |
| Real DESED `validation.tsv` | 1168 clips, 4239 strong events, made of `eval_dcase2018.tsv` plus `test_dcase2018.tsv` | The waves are not on disk. This list is real-data validation, not a second copy of the synthetic validation split. |
| `weak.tsv` | 1578 real clips with clip-level tags and no times | The waves are not on disk, and a clip tag does not say when the sound happens. |
| `unlabel_in_domain.tsv` | 14412 filenames and no labels | There is nothing to supervise, and the waves are not on disk. |
| `audioset_strong.tsv` | 15446 extra strong labels on 3470 clips | The waves are not on disk. The names overlap the weak list, the unlabeled list, and real validation, so this file is extra annotation, not its own split. |
| Freesound Dataset 50K (FSD50K) | Tables for 200 classes and clip tags. Development and evaluation splits are present as tables. No waves | FSD50K has no onset or offset. It is a vocabulary source for later street classes, and it does not enter this pipeline. |

The 2019 synthetic folder is the DCASE 2019 synthetic evaluation set.

## 4. Current pipeline

The steps below are what `python -m nndl_project.baseline` does.

1. Read the training wav names from `data/raw/desed/dcase_synth/audio/train/synthetic21_train/soundscapes/`. Read strong labels from `metadata/train/synthetic21_train/soundscapes.tsv`. Ignore the `.jams` recipe files and the `.txt` copy of the labels.
2. Load each wav as one channel and resample it to 16000 Hz and 10 seconds.
3. Compute the log-mel matrix, shape `(497, 64)`.
4. Build the frame targets, shape `(497, 10)`, from the strong labels of that same split.
5. Cache both arrays under `data/cache`. The cache is gitignored with the rest of `data/*`.
6. Standardize each mel band with the mean and spread of the training frames.
7. Fit ten logistic regressions, one per class, with scikit-learn on the central processing unit (CPU).
8. Repeat steps 2 to 5 for `synthetic21_validation`. Run the fitted models to get frame probabilities.
9. Smooth each class along time with a median filter of 11 frames (about 220 milliseconds). Keep frames at or above 0.5.
10. Turn each run of active frames into an event. The score is the mean smoothed probability inside the run. The boundaries sit on the 20 millisecond hop grid.
11. Compare those events with the validation strong labels. Report segment F1 and event F1, per class, plus micro and macro. F1 means the harmonic mean of precision and recall. The two definitions are written out after the diagram.

```text
synthetic21_train wav + soundscapes.tsv
    -> one channel at 16000 Hz
    -> log-mel (497, 64)
    -> frame targets (497, 10)
    -> ten logistic regressions

synthetic21_validation wav
    -> same log-mel
    -> frame probabilities (497, 10)
    -> median filter of 11 frames, threshold 0.5
    -> events: label, start_sec, end_sec, score
    -> segment F1 and event F1
       against synthetic21_validation soundscapes.tsv
```

Segment F1 uses bins of 1 second. A bin is positive for a class when any event of that class overlaps the bin. The same rule is applied to the reference labels and to the system events. Precision, recall, and F1 are computed from the pooled counts. F1 is the harmonic mean of precision and recall. Micro F1 pools true positives, false positives, and false negatives across classes. Macro F1 averages the per-class F1 scores that are defined.

Event F1 matches one prediction to one reference of the same class, inside one clip. A match requires the start times to fall within 200 milliseconds, and the end times to fall within the larger of 200 milliseconds and 20 percent of the reference duration. A reference event can match only one prediction.

## 5. Logistic baseline

The current model is ten independent logistic regressions. Class `c` sees one log-mel row of length 64 and outputs one probability. There is no shared hidden layer and no look at neighboring frames.

Training uses scikit-learn `SGDClassifier` with log loss. Log loss is the classification loss for a yes/no probability. SGD means stochastic gradient descent. The penalty is L2, a penalty on the squared size of the weights, with alpha 0.0001. Weights are averaged. There are 5 epochs and batches of 32 clips. The positive and negative frame weights are computed once from the whole training set, so a rare class is not ignored. The threshold 0.5 was not tuned on validation.

On all 2500 `synthetic21_validation` clips the scores are:

| Score | Micro F1 | Macro F1 |
| --- | ---: | ---: |
| Segment, 1 second bins | 0.393 | 0.289 |
| Event, 200 millisecond collar | 0.044 | 0.030 |

Speech has the highest segment F1, 0.778. Event F1 is much lower because the model emits many extra short events and the edges are loose. A one-frame classifier has no context, so that gap is the reason for the next model. The per-class counts are in `reports/baseline_logistic.md`. The file `reports/baseline_logistic_smoke.md` is a 200-clip pipeline check. Those smoke numbers are not the baseline.

## 6. Next model: context multilayer perceptron

The model after the logistic baseline is one shared multilayer perceptron (MLP). An MLP is a stack of dense layers. This one is not a convolutional network, and it does not take the log-mel matrix as an image. It has been trained. Section 9 records the scores. Section 10 is the network that comes after it.

Each example is the frame being classified, the 5 frames before it, and the 5 frames after it. That is 11 frames by 64 bands, flattened to 704 numbers, about 220 milliseconds of context. The edges of a clip repeat the first or last frame so every one of the 497 frames still has a full window.

The layers are:

1. Linear layer from 704 to 128.
2. Rectified linear unit (ReLU). ReLU keeps positive values and sets negative values to zero.
3. Linear layer from 128 to 10.
4. A sigmoid on each of the 10 outputs, so each class has its own probability and classes can overlap.

One network is shared across classes. The logistic baseline trained ten separate weight vectors. The MLP can use the fact that classes co-occur, while the output head still allows overlap.

The following pieces stay fixed, so the comparison with section 5 is fair.

- The same log-mel features, reused from `data/cache`. The 704-number window is built while training. The full matrix of windows is not written to disk.
- The same frame targets.
- The same class weights idea: a positive frame of a rare class counts more than a negative frame. The weights come from the training counts.
- The same decode: median filter of 11 frames, then threshold 0.5.
- The same two scores, on all 2500 `synthetic21_validation` clips.
- The same closed sets. Public eval and the 16 `dcase2019` folders stay closed.

The first run is a 200-clip smoke check, so a broken path fails before the full 10000 clips. The full run is the one that may be compared with segment micro F1 0.393, segment macro F1 0.289, event micro F1 0.044, and event macro F1 0.030. Threshold 0.5 stays in place for that comparison.

## 7. Framework: PyTorch

This project uses PyTorch for the deep learning models.

TensorFlow can train the same MLP. PyTorch is the framework we will write the model in, for four reasons that match this repository.

1. Training has to read the cached log-mel arrays and build an 11-frame window in ordinary Python. PyTorch eager mode is that kind of loop: each batch is a normal tensor operation.
2. This machine has no graphics processing unit (GPU). A graph runtime does not help a small dense net on CPU.
3. The features already exist as NumPy arrays. PyTorch can take those arrays directly. The log-mel code we already have stays the feature front end, so the training code does not need torchaudio and does not need a TensorFlow signal stack.
4. Published DESED and DCASE training code is usually PyTorch. A later comparison with a paper setup can stay in one framework.

The course constraint is about image-style convolutional models. It is not a constraint on the library brand. The context MLP is a small dense net on rows of the log-mel matrix, written in PyTorch.

## 8. How to run the context MLP

The training script is `src/nndl_project/train_context_mlp.py`. Run it from the project directory in a terminal you can watch. Stop it with Ctrl+C. The last epoch that finished stays on disk.

Smoke check, 200 training clips and 200 validation clips:

`uv run python -m nndl_project.train_context_mlp --limit 200`

Full training, all 10000 training clips and all 2500 validation clips:

`uv run python -m nndl_project.train_context_mlp`

Each batch prints the weighted loss, the running average, and an estimate of the time left in that pass. At the end of every epoch the script prints segment F1 and event F1 for every class, then writes the checkpoint and the report again. The saved model is the last finished epoch. Validation is not used to pick a different epoch, and the threshold stays at 0.5.

A full run writes `reports/context_mlp.md` and `reports/context_mlp.json`. A prefix run writes `reports/context_mlp_smoke.md` and `reports/context_mlp_smoke.json` instead, so it does not replace the full-data score. Checkpoints and the plain text log are under `data/cache/context_mlp/`, which git ignores along with the rest of `data/*`. Continue a stopped full run with:

`uv run python -m nndl_project.train_context_mlp --resume`

Raise `--epochs` above the last finished epoch when you resume, or the script will tell you there is nothing left to do.

## 9. What the context MLP showed

The context MLP has been trained. The run used all 10000 synthetic21_train clips and all 2500 synthetic21_validation clips. It was interrupted after epoch 432 of a request for 1000 epochs. The scores in `reports/context_mlp.md` are that last finished epoch.

| Score | Logistic baseline | Context MLP epoch 432 |
| --- | ---: | ---: |
| Segment micro F1 | 0.393 | 0.398 |
| Segment macro F1 | 0.289 | 0.349 |
| Event micro F1 | 0.044 | 0.053 |
| Event macro F1 | 0.030 | 0.041 |

Segment macro F1 rose. Segment micro F1 stayed near the baseline. Event F1 stayed near 0.05 for both models. Training loss kept falling through epoch 432. Validation segment micro F1 was already near the baseline by about epoch 50, and later epochs wander between about 0.37 and 0.41. More epochs of the same MLP will not fix the event times.

The MLP sees about 220 milliseconds. On validation clip `0.wav`, reference Speech runs from 6.832 s to 8.674 s. The MLP's main Speech region starts at 7.240 s, about 400 milliseconds late, so it misses the 200 millisecond onset collar, and Speech is split into several short events. The next network is there to carry an event across the whole clip.

The file `reports/context_mlp_smoke.md` is a 200-clip pipeline check. Those smoke numbers are not this comparison.

## 10. Next model: frame gated recurrent unit

The next model is one shared bidirectional GRU. A GRU keeps a hidden state and updates it once per frame. Bidirectional means one state reads the clip forward and one state reads it backward. The two states are concatenated. This is a sequence model on the log-mel frames. It does not take the log-mel matrix as an image, and it has no convolutional layer.

The input is one clip, shape (497, 64), after the same per-band training mean and standard deviation the MLP used. A batch is (batch, 497, 64). The frames stay in time order. They are not flattened into independent rows.

The layers are:

1. One bidirectional GRU. Input size 64, hidden size 64 in each direction, one layer. Each frame then has 128 numbers.
2. One linear layer from 128 to 10 logits.
3. A sigmoid on each class, applied inside binary cross-entropy on the logits. Classes can overlap.

The parameter count is 51210, which is smaller than the context MLP. There is no dropout. The optimizer is Adam, learning rate 0.001, weight decay 0, batch 16 clips, seed 0. After backward, the global gradient norm is clipped at 5, and the logged grad_norm is the norm before that clip. The positive class weights are the same idea as the MLP: negative frames divided by positive frames.

The following pieces stay fixed, so a change in event F1 can be attributed to the longer memory.

- The same log-mel cache. The script reuses a finished cache and does not write a second copy of the matrices.
- The same frame targets.
- The same decode: median filter of 11 frames, then threshold 0.5. The threshold is not chosen on validation.
- The same two scores, on all 2500 `synthetic21_validation` clips.
- The same closed sets. Public eval and the 16 `dcase2019` folders stay closed.
- The designated model is the last finished epoch. Validation F1 is printed for monitoring. It does not pick the checkpoint and it does not stop the run.

The default request is 30 epochs. That is the planned run. If validation loss is still falling at epoch 30, resume with a higher `--epochs`. Do not resume the context MLP toward 1000 epochs for this comparison.

The training script is `src/nndl_project/train_frame_gru.py`. Run it from the project directory in a terminal you can watch. Stop it with Ctrl+C. The epoch in progress is dropped. The last epoch that finished stays on disk.

Smoke check, 200 training clips and 200 validation clips:

`uv run python -m nndl_project.train_frame_gru --limit 200`

Full training, all 10000 training clips and all 2500 validation clips:

`uv run python -m nndl_project.train_frame_gru`

Continue a stopped full run. `--epochs` must be greater than the last finished epoch:

`uv run python -m nndl_project.train_frame_gru --resume --epochs 30`

Each batch prints the weighted loss, the running average, the unweighted average, the gradient norm on a training batch, and an estimate of the time left in that pass. At the end of every epoch the script prints segment F1 and event F1 for every class, then writes the checkpoint and the report again.

A full planned run writes `reports/frame_gru.md` and `reports/frame_gru.json`. A prefix run writes `reports/frame_gru_smoke.md` and `reports/frame_gru_smoke.json`, so it does not replace the full-data score. The planned checkpoints go to `data/cache/frame_gru/full`, which git ignores along with the rest of `data/*`. A different hidden size, learning rate, weight decay, gradient clip, batch size, or seed is stored in its own folder and its own report name, so it cannot replace the planned checkpoint. The context MLP checkpoints under `data/cache/context_mlp/full` are left as they are.

One epoch walks 497 recurrent steps per clip on the CPU. The time-left line on the first epoch is the estimate to trust. If that line says more than about 15 minutes for the epoch, stop and rerun with a smaller `--hidden` before leaving the job overnight.

## 11. What is not done yet

The frame GRU script is written. It has not been trained on the full split yet. The decision threshold has not been tuned. Public eval, the `dcase2019` conditions, the real DESED lists, and FSD50K have not been scored. No wav files, feature caches, or archives are part of the git history.
