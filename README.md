# Urban sound event detection with timestamps

Neural Networks and Deep Learning project. Deeptha Dinakar (J010) and Gaurang Gupta (J013).

Input: a ~10 second, one-channel recording. Output: a list of events, each with a class name, a start time in seconds, an end time in seconds, and a score between 0 and 1. More than one sound can be active at once.

Output format (illustrative numbers):

```text
$ uv run python -m nndl_project.predict clip.wav --street
clip.wav  (3 events, model small_crnn epoch 10)
  speech_crowd                   0.000 s ->   2.120 s   score 0.871
  other                          3.780 s ->  10.000 s   score 0.744
  dog                            6.960 s ->   7.900 s   score 0.776
```

## Results on synthetic validation (2500 clips, fixed decode: median 11 frames, threshold 0.5)

| Model | Segment micro F1 | Segment macro F1 | Event micro F1 | Event macro F1 |
|---|---:|---:|---:|---:|
| Logistic regression per frame (baseline) | 0.393 | 0.289 | 0.044 | 0.030 |
| Context MLP, 11-frame window | 0.398 | 0.349 | 0.053 | 0.041 |
| Bidirectional GRU | 0.509 | 0.438 | 0.104 | 0.096 |
| Small CRNN (2 conv + BiGRU) | **0.612** | **0.542** | **0.142** | **0.155** |

Held-out scores (real public test, DCASE 2019 synthetic conditions) and the tuned decode come from `nndl_project.evaluate`; see `reports/final_evaluation_crnn.md` after it runs. The full write-up is `reports/final_report.md`.

## Layout

| Path | What it is |
|---|---|
| `src/nndl_project/constants.py` | Class names, feature settings, decode and metric settings |
| `src/nndl_project/data.py` | Strong-label reader, split paths, frame targets |
| `src/nndl_project/features.py` | Wav loading (mono, 16 kHz, 10 s) and log-mel (497 x 64) |
| `src/nndl_project/decode.py` | Frame probabilities to events (median filter + threshold) |
| `src/nndl_project/metrics.py` | Segment F1 (1 s bins) and event F1 (200 ms collar) |
| `src/nndl_project/label_map.py` | The fixed ten street classes and the DESED-to-street map |
| `src/nndl_project/baseline.py` | Logistic baseline, feature cache |
| `src/nndl_project/context.py`, `train_context_mlp.py` | Context MLP |
| `src/nndl_project/sequence.py`, `train_frame_gru.py` | Bidirectional GRU |
| `src/nndl_project/crnn.py`, `train_crnn.py` | Small CRNN |
| `src/nndl_project/predict.py` | Wav in, events out (the project output) |
| `src/nndl_project/evaluate.py` | Tune the decode on validation, score public test and DCASE 2019 |
| `reports/` | One report per trained model, the design note, the final report |
| `notebooks/` | Data audit, and a per-clip comparison of the four models |
| `tests/` | Splits, frames and metrics, models, label map, decode, prediction |

## Setup

```powershell
uv sync
dvc remote modify --local storage sas_token "PASTE_THE_SAS_TOKEN_HERE"
dvc pull -j 2          # data/raw, about 43 GB
uv run pytest
```

See `DVC.md` for the storage setup. Only `data/raw` is tracked by DVC in the current commit. If `data/cache.dvc` is not in your checkout, the trained checkpoints are not in the remote: copy `data/cache/crnn/full/model_last.pt` from the machine that trained it, or retrain.

## Commands

```powershell
uv run python -m nndl_project.baseline              # logistic baseline (~12 min)
uv run python -m nndl_project.train_context_mlp     # context MLP
uv run python -m nndl_project.train_frame_gru       # GRU, 30 epochs (~12 min each on CPU)
uv run python -m nndl_project.train_crnn            # CRNN, 10 epochs (~16 min each on CPU)
uv run python -m nndl_project.evaluate --quick      # tune on validation, score public test + fbsnr_30dB
uv run python -m nndl_project.evaluate              # all 16 DCASE 2019 conditions as well
uv run python -m nndl_project.predict clip.wav --street --thresholds reports/final_evaluation_crnn.json
```

Every training script takes `--limit 200` for a smoke check and `--resume` to continue.

## Data rules

Training uses only `synthetic21_train` (10,000 clips). Validation, and the only place anything is tuned, is `synthetic21_validation` (2,500 clips). The DESED public test (692 real clips) and the 16 DCASE 2019 condition folders are scored once, at the end, with settings frozen on validation. Official splits are kept as published.
