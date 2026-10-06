"""Train the frame GRU and save every finished epoch.

Run this yourself from the project directory, in a terminal you can watch
and interrupt. Do not resume the context MLP for this experiment. That run
already flattened, and its checkpoints live in a different folder.

Smoke check, 200 training clips and 200 validation clips:

    uv run python -m nndl_project.train_frame_gru --limit 200

Full comparison, all 10000 synthetic21_train clips and all 2500
synthetic21_validation clips. This is the command whose scores may be
compared with the logistic baseline and with context MLP epoch 432:

    uv run python -m nndl_project.train_frame_gru

Continue a stopped run. Raise --epochs above the last finished epoch:

    uv run python -m nndl_project.train_frame_gru --resume --epochs 30

What this script does, in order:

1. Loads the cached log-mel matrices. It does not recompute them when the
   cache meta already matches, and it does not write a second copy.
2. Standardizes each mel band with the mean and standard deviation of the
   training frames only. Validation frames use those same training numbers.
3. Trains one bidirectional gated recurrent unit (GRU) with Adam. The loss
   is binary cross-entropy on logits, with a positive weight per class.
4. After each epoch, scores the validation clips with the same decode as the
   other two models: median filter of 11 frames, then threshold 0.5.
5. Writes the checkpoint and the report before the next epoch starts.

The designated model is the last epoch that finished. Validation F1 is
printed so you can watch the run. It is not used to pick a better epoch, to
stop early, or to move the threshold. F1 is the harmonic mean of precision
and recall.

Ctrl+C stops the run. The epoch that was in progress is discarded. The last
epoch that had already been written stays on disk. The process exits 130.

Public eval and the dcase2019 folders are not read. This script only asks
for the synthetic21_train and synthetic21_validation splits.

One epoch walks 497 recurrent steps per clip, on the CPU. The first epoch's
"eta" line is the time estimate to trust. If that line says the epoch will
take more than about 15 minutes, stop and rerun with a smaller --hidden.
A different hidden size is stored in its own folder. It will not resume the
hidden-size-64 checkpoint, and it will not replace that checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from torch import nn

from nndl_project.baseline import (
    EXPECTED_CLIPS,
    build_feature_cache,
    feature_moments,
    positive_frame_counts,
    resolve_jobs,
    score_split,
    select_clips,
)
from nndl_project.constants import (
    CLASSES,
    FEATURE_VERSION,
    HOP,
    MEDIAN_FRAMES,
    N_FFT,
    N_FRAMES,
    N_MELS,
    SAMPLE_RATE,
    THRESHOLD,
)
from nndl_project.data import Event, project_root
from nndl_project.metrics import Counts, macro_f1, micro_counts, precision_recall_f1
from nndl_project.sequence import BIDIRECTIONAL, HIDDEN_UNITS, LAYERS, FrameGRU

# 30 epochs, not 1000. The context MLP's validation scores had already
# flattened by about epoch 50, while its training loss kept falling. The
# default here is long enough to see whether the GRU is learning, and short
# enough that a CPU run can finish in one sitting if each epoch is a few
# minutes. Raise --epochs and pass --resume if the validation loss is still
# falling when epoch 30 finishes.
DEFAULT_EPOCHS = 30
DEFAULT_BATCH_CLIPS = 16
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 0.0
# Global gradient norm cap. A recurrent net can produce one huge step even
# when most steps are small. Clipping that step keeps a long CPU run from
# ending in NaN. The clip value is part of the run identity.
DEFAULT_GRAD_CLIP = 5.0

# These two blocks are copied from the saved reports. They are references for
# the log and the markdown file. They are not recomputed, and this script
# does not load either model's checkpoints.
LOGISTIC_REFERENCE = {
    "segment_micro_f1": 0.393,
    "segment_macro_f1": 0.289,
    "event_micro_f1": 0.044,
    "event_macro_f1": 0.030,
    "split": "all 2500 synthetic21_validation clips",
    "note": (
        "Frame-wise logistic regression with threshold 0.5. "
        "Compare a frame GRU run with these numbers only when that run also scores all 2500 validation clips."
    ),
}
CONTEXT_MLP_REFERENCE = {
    "epoch": 432,
    "epochs_requested": 1000,
    "interrupted": True,
    "segment_micro_f1": 0.398,
    "segment_macro_f1": 0.349,
    "event_micro_f1": 0.053,
    "event_macro_f1": 0.041,
    "split": "all 2500 synthetic21_validation clips",
    "note": (
        "Last finished epoch of the context MLP, not a best checkpoint. "
        "Compare a frame GRU run with these numbers only when that run also scores all 2500 validation clips."
    ),
}


class RunLog:
    """Print each line immediately and append the same line to train.log.

    The handle is flushed on every line so a crash or Ctrl+C still leaves
    the lines that were already printed. The file is opened in append mode
    so a resumed run keeps the earlier log.
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._handle = path.open("a", encoding="utf-8", newline="\n")

    def line(self, text: str = "") -> None:
        print(text, flush=True)
        self._handle.write(text + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _counts_payload(by_class: dict[str, Counts]) -> dict[str, object]:
    """Turn pooled counts into JSON-safe precision, recall, and F1.

    A class with no true positives, false positives, or false negatives has
    nothing to score. Its precision, recall, and F1 are null, and macro F1
    skips that class. Micro F1 pools the counts first, then computes one F1.
    """
    classes: dict[str, object] = {}
    for name in CLASSES:
        counts = by_class[name]
        row = precision_recall_f1(counts)
        classes[name] = {
            "tp": counts.tp,
            "fp": counts.fp,
            "fn": counts.fn,
            "precision": None if row is None else row[0],
            "recall": None if row is None else row[1],
            "f1": None if row is None else row[2],
        }
    micro = micro_counts(by_class)
    micro_row = precision_recall_f1(micro)
    macro = macro_f1(by_class)
    return {
        "classes": classes,
        "micro": {
            "tp": micro.tp,
            "fp": micro.fp,
            "fn": micro.fn,
            "precision": None if micro_row is None else micro_row[0],
            "recall": None if micro_row is None else micro_row[1],
            "f1": None if micro_row is None else micro_row[2],
        },
        "macro_f1": macro,
    }


def _metric_table(payload: dict[str, object]) -> str:
    lines = [
        "| class | precision | recall | f1 | tp | fp | fn |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    classes = payload["classes"]
    assert isinstance(classes, dict)
    for name in CLASSES:
        row = classes[name]
        assert isinstance(row, dict)
        lines.append(
            f"| {name} | {_fmt(row['precision'])} | {_fmt(row['recall'])} | {_fmt(row['f1'])} | "
            f"{row['tp']} | {row['fp']} | {row['fn']} |"
        )
    micro = payload["micro"]
    assert isinstance(micro, dict)
    lines.append(
        f"| micro | {_fmt(micro['precision'])} | {_fmt(micro['recall'])} | {_fmt(micro['f1'])} | "
        f"{micro['tp']} | {micro['fp']} | {micro['fn']} |"
    )
    lines.append(f"| macro f1 |  |  | {_fmt(payload['macro_f1'])} |  |  |  |")
    return "\n".join(lines)


def _fmt(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def _event_lines(events: list[Event]) -> str:
    if not events:
        return "None."
    lines = []
    for event in events:
        score = "" if event.score is None else f", score {event.score:.3f}"
        lines.append(f"- {event.label} from {event.onset:.3f} s to {event.offset:.3f} s{score}")
    return "\n".join(lines)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    """Write JSON by replacing a finished temp file, so a crash cannot leave a half file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _pos_weight(positives: np.ndarray, n_frames: int) -> list[float]:
    """Positive weight = negative frames / positive frames, for each class.

    Rare classes would otherwise contribute almost nothing to an unweighted
    mean, because almost every frame is negative. The same formula was used
    for the context MLP. A class with no positive frames, or no negative
    frames, gets weight 1 so the loss stays defined.
    """
    weights: list[float] = []
    for n_pos in positives:
        n_neg = n_frames - int(n_pos)
        if int(n_pos) <= 0 or n_neg <= 0:
            weights.append(1.0)
        else:
            weights.append(n_neg / float(n_pos))
    return weights


def _planned_settings(args: argparse.Namespace) -> bool:
    """True only for the one run we will compare with the MLP and the logistic model."""
    return (
        args.hidden == HIDDEN_UNITS
        and args.lr == DEFAULT_LR
        and args.weight_decay == DEFAULT_WEIGHT_DECAY
        and args.grad_clip == DEFAULT_GRAD_CLIP
        and args.batch_clips == DEFAULT_BATCH_CLIPS
        and args.seed == 0
    )


def _settings_suffix(args: argparse.Namespace) -> str:
    """Folder tag for a run that is not the planned comparison.

    The planned full run lives in data/cache/frame_gru/full. A smaller hidden
    size, or any other changed setting, gets its own folder so it cannot
    replace those checkpoints. The tag is empty when every setting matches.
    """
    parts: list[str] = []
    if args.hidden != HIDDEN_UNITS:
        parts.append(f"h{args.hidden}")
    if args.lr != DEFAULT_LR:
        parts.append(f"lr{args.lr:g}")
    if args.weight_decay != DEFAULT_WEIGHT_DECAY:
        parts.append(f"wd{args.weight_decay:g}")
    if args.grad_clip != DEFAULT_GRAD_CLIP:
        parts.append(f"clip{args.grad_clip:g}")
    if args.batch_clips != DEFAULT_BATCH_CLIPS:
        parts.append(f"batch{args.batch_clips}")
    if args.seed != 0:
        parts.append(f"seed{args.seed}")
    return "_".join(parts)


def _run_dir(smoke: bool, train_count: int, val_count: int, args: argparse.Namespace) -> Path:
    name = f"limit{train_count}_val{val_count}" if smoke else "full"
    suffix = _settings_suffix(args)
    if suffix:
        name = f"{name}_{suffix}"
    return project_root() / "data" / "cache" / "frame_gru" / name


def _report_paths(smoke: bool, args: argparse.Namespace) -> tuple[Path, Path]:
    """Pick report names that cannot overwrite the full planned comparison.

    A prefix always writes the smoke names, including a 2-clip check.
    The planned 10000/2500 run writes reports/frame_gru.md and .json.
    Any other full-split setting writes a name that includes its suffix.
    """
    reports = project_root() / "reports"
    if smoke:
        return reports / "frame_gru_smoke.md", reports / "frame_gru_smoke.json"
    suffix = _settings_suffix(args)
    if suffix:
        return reports / f"frame_gru_{suffix}.md", reports / f"frame_gru_{suffix}.json"
    return reports / "frame_gru.md", reports / "frame_gru.json"


def _identity(args: argparse.Namespace, train_count: int, val_count: int) -> dict[str, object]:
    """Settings that must match before a checkpoint may be resumed.

    epochs_requested is left out on purpose. A stopped run is continued by
    passing --resume and a larger --epochs. Hidden size, depth, direction,
    learning rate, weight decay, gradient clip, batch size, seed, and the
    clip counts are part of the experiment. A mismatch refuses to resume
    rather than training a different model into an old checkpoint.
    """
    return {
        "model": "frame_gru",
        "hidden": args.hidden,
        "layers": LAYERS,
        "bidirectional": BIDIRECTIONAL,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "batch_clips": args.batch_clips,
        "seed": args.seed,
        "train_clips": train_count,
        "val_clips": val_count,
    }


def _load_resume(path: Path, identity: dict[str, object], log: RunLog) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"--resume was set, but there is no checkpoint at {path}")
    # weights_only is False because the checkpoint stores the record and the
    # identity dict as well as the tensors. These files are written by this
    # script in the project cache. Do not load a checkpoint from anywhere else.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    saved = payload.get("identity")
    if saved != identity:
        raise RuntimeError(
            "The checkpoint was trained with different settings. "
            f"Saved {saved}. This command is {identity}. "
            "Start a new run, or repeat the same settings."
        )
    log.line(f"resuming from epoch {payload['epoch']} in {path}")
    return payload


def _normalize(raw: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Apply the training mean and standard deviation to a batch of clips.

    raw has shape (batch, frames, bands). mean and std have shape (bands,).
    The same numbers are used for training and validation. Fitting them on
    validation would leak the split we report.
    """
    return ((raw - mean) / std).astype(np.float32, copy=False)


def _forward_epoch(
    model: FrameGRU,
    features: np.memmap,
    targets: np.memmap,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    train: bool,
    optimizer: torch.optim.Optimizer | None,
    criterion: nn.BCEWithLogitsLoss,
    unweighted: nn.BCEWithLogitsLoss,
    batch_clips: int,
    grad_clip: float,
    device: torch.device,
    log: RunLog,
    epoch_label: str,
    generator: np.random.Generator,
) -> tuple[np.ndarray | None, float, float]:
    """Run one pass over every clip.

    Training shuffles clips with `generator` and updates the weights.
    Validation walks clips in stored order, does not update weights, and
    returns a probability array aligned with that stored order.

    The time axis stays intact. A batch tensor is (batch, 497, 64), and the
    labels are (batch, 497, 10). The loss averages every frame and every
    class. The epoch average weights each batch by how many label elements
    it had, so a short last batch does not count as much as a full one.

    Probabilities for validation are written with probabilities[batch_ids].
    A sequential slice would be wrong whenever the order is not 0, 1, 2, ...
    Training does not shuffle on this path, but the index write keeps the
    alignment obvious and safe.
    """
    n_clips = int(features.shape[0])
    if train:
        assert optimizer is not None
        model.train()
        order = generator.permutation(n_clips)
    else:
        # This run has no dropout. eval() is still set so a later dropout
        # would not leak into the validation score by accident.
        model.eval()
        order = np.arange(n_clips)
        probabilities = np.empty((n_clips, N_FRAMES, len(CLASSES)), dtype=np.float32)
    weighted_sum = 0.0
    unweighted_sum = 0.0
    seen = 0
    started = time.perf_counter()
    n_batches = int(np.ceil(n_clips / batch_clips))
    # Validation must not build a graph. 2500 clips of length 497 would hold
    # that graph for the whole pass.
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch_index, start in enumerate(range(0, n_clips, batch_clips), start=1):
            batch_ids = order[start : start + batch_clips]
            raw = np.asarray(features[batch_ids], dtype=np.float32)
            design = torch.from_numpy(_normalize(raw, mean, std)).to(device)
            labels_np = np.asarray(targets[batch_ids], dtype=np.float32)
            labels = torch.from_numpy(labels_np).to(device)
            if design.shape != (raw.shape[0], N_FRAMES, N_MELS):
                raise RuntimeError(f"expected features {(raw.shape[0], N_FRAMES, N_MELS)}, got {tuple(design.shape)}")
            if labels.shape != (raw.shape[0], N_FRAMES, len(CLASSES)):
                raise RuntimeError(f"expected labels {(raw.shape[0], N_FRAMES, len(CLASSES))}, got {tuple(labels.shape)}")
            logits = model(design)
            weighted_loss = criterion(logits, labels)
            plain_loss = unweighted(logits, labels)
            grad_norm = 0.0
            if train:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                weighted_loss.backward()
                # Clip, then read the returned total norm. The returned value
                # is the norm BEFORE the clip, so a number above grad_clip
                # means this step was shortened.
                grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip))
                optimizer.step()
            else:
                probabilities[batch_ids] = torch.sigmoid(logits).detach().cpu().numpy()
            elements = int(labels.numel())
            weighted_sum += float(weighted_loss.item()) * elements
            unweighted_sum += float(plain_loss.item()) * elements
            seen += elements
            elapsed = time.perf_counter() - started
            clips_done = start + int(raw.shape[0])
            rate = clips_done / elapsed if elapsed > 0 else 0.0
            remaining = (n_clips - clips_done) / rate if rate > 0 else 0.0
            grad_text = f" grad_norm {grad_norm:.3f}" if train else ""
            log.line(
                f"{epoch_label} batch {batch_index}/{n_batches} "
                f"clips {clips_done}/{n_clips} "
                f"weighted loss {weighted_loss.item():.4f} "
                f"epoch avg {weighted_sum / seen:.4f} "
                f"unweighted avg {unweighted_sum / seen:.4f}"
                f"{grad_text} "
                f"{rate:.1f} clips/s eta {remaining / 60:.1f} min"
            )
    if seen == 0:
        raise RuntimeError("no frames were scored")
    if train:
        return None, weighted_sum / seen, unweighted_sum / seen
    return probabilities, weighted_sum / seen, unweighted_sum / seen


def _render_report(path: Path, record: dict[str, object]) -> None:
    epochs = record["epochs"]
    assert isinstance(epochs, list)
    if record["smoke"]:
        opening = (
            "This file is a pipeline check on a prefix of the synthetic clips. "
            "It is not the full-data frame GRU score."
        )
    elif not record["planned_settings"]:
        opening = (
            "This file is a full-split run with settings that differ from the planned comparison. "
            "It is not the hidden-size-64 frame GRU score."
        )
    elif record["interrupted"] and not epochs:
        opening = "The run was interrupted before the first epoch finished. There is no score yet."
    elif record["interrupted"]:
        opening = (
            "The run was interrupted. The scores below are from the last epoch that finished. "
            "They are not a cherry-picked checkpoint."
        )
    else:
        opening = (
            "These scores are the frame gated recurrent unit on the clips named below. "
            "The model is the last finished epoch. The threshold was not tuned."
        )
    history = [
        "| epoch | train weighted loss | train unweighted loss | val weighted loss | val unweighted loss | segment micro f1 | segment macro f1 | event micro f1 | event macro f1 | seconds |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in epochs:
        assert isinstance(row, dict)
        segment = row["segment"]
        event = row["event"]
        assert isinstance(segment, dict) and isinstance(event, dict)
        history.append(
            f"| {row['epoch']} | {row['train_loss_weighted']:.4f} | {row['train_loss_unweighted']:.4f} | "
            f"{row['val_loss_weighted']:.4f} | {row['val_loss_unweighted']:.4f} | "
            f"{_fmt(segment['micro']['f1'])} | {_fmt(segment['macro_f1'])} | "
            f"{_fmt(event['micro']['f1'])} | {_fmt(event['macro_f1'])} | {row['seconds']:.1f} |"
        )
    last_tables = "No epoch finished."
    if epochs:
        last = epochs[-1]
        assert isinstance(last, dict)
        last_tables = (
            f"Epoch {last['epoch']}.\n\n"
            f"Segment scores\n\n{_metric_table(last['segment'])}\n\n"
            f"Event scores\n\n{_metric_table(last['event'])}"
        )
    weights = record["pos_weight"]
    assert isinstance(weights, list)
    weight_text = ", ".join(f"{name} {weight:.2f}" for name, weight in zip(CLASSES, weights, strict=True))
    logistic = LOGISTIC_REFERENCE
    mlp = CONTEXT_MLP_REFERENCE
    text = f"""# Frame gated recurrent unit

{opening}

Command: `{record["command"]}`

Started: {record["started_at"]}
Updated: {record["updated_at"]}
Interrupted: {record["interrupted"]}
Device: {record["device"]}
Project root: `{record["project_root"]}`

## Setup

- Train clips: {record["train_clips"]}, {record["train_first"]} through {record["train_last"]}.
- Validation clips: {record["val_clips"]}, {record["val_first"]} through {record["val_last"]}.
- Network: bidirectional GRU, input 64, hidden {record["hidden"]} in each direction, {record["layers"]} layer, then linear {record["hidden_concat"]} to 10 logits. Sigmoid per class, inside the loss.
- Parameters: {record["n_parameters"]}.
- Loss: binary cross-entropy on logits. Positive weight for a class is negative frames divided by positive frames: {weight_text}.
- Optimizer: Adam, learning rate {record["learning_rate"]}, weight decay {record["weight_decay"]}, batch {record["batch_clips"]} clips, seed {record["seed"]}.
- Gradient clip: global norm {record["grad_clip"]}, applied after backward and before the Adam step. The logged grad_norm is the norm before that clip.
- Decode: median filter of {record["median_frames"]} frames, threshold {record["threshold"]}. The threshold was not chosen on validation.
- Designated model: the last finished epoch. Validation F1 is not used to pick a checkpoint.
- Feature cache version {record["feature_version"]}: {record["n_mels"]} mel bands, FFT {record["n_fft"]}, hop {record["hop"]} at {record["sample_rate"]} Hz, {record["n_frames"]} frames.
- Standardization: training-set mean and standard deviation per mel band.
- PyTorch {record["torch_version"]}, NumPy {record["numpy_version"]}.
- Checkpoint: `{record["checkpoint"]}`.
- Log: `{record["log_path"]}`.
- Epochs requested: {record["epochs_requested"]}. Epochs finished: {record["epochs_completed"]}.

Logistic baseline on all 2500 validation clips: segment micro F1 {logistic["segment_micro_f1"]}, segment macro F1 {logistic["segment_macro_f1"]}, event micro F1 {logistic["event_micro_f1"]}, event macro F1 {logistic["event_macro_f1"]}.

Context MLP epoch {mlp["epoch"]} of {mlp["epochs_requested"]}, interrupted, on all 2500 validation clips: segment micro F1 {mlp["segment_micro_f1"]}, segment macro F1 {mlp["segment_macro_f1"]}, event micro F1 {mlp["event_micro_f1"]}, event macro F1 {mlp["event_macro_f1"]}.

Compare this file with those two lines only when this run also scores all 2500 validation clips and uses the planned settings (hidden 64, learning rate 0.001, weight decay 0, gradient clip 5, batch 16, seed 0).

## First validation clip

Clip: {record["first_name"]}

Reference labels:

{record["first_reference_text"]}

Model events from the last finished epoch:

{record["first_prediction_text"]}

## Epoch history

{chr(10).join(history)}

## Last finished epoch

{last_tables}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _save_epoch(
    record: dict[str, object],
    model: FrameGRU,
    optimizer: torch.optim.Optimizer,
    mean: np.ndarray,
    std: np.ndarray,
    identity: dict[str, object],
    run_dir: Path,
    report_md: Path,
    report_json: Path,
    epoch: int,
) -> None:
    """Save the epoch before the next one starts.

    epoch_XXX.pt is that epoch. model_last.pt is always a copy of the newest
    finished epoch, which is the designated model. Both files hold the
    weights, the Adam state, the standardization, the identity, and the full
    record, so a resume does not depend on a side file.

    The report is rewritten from scratch each epoch, including the history of
    every finished epoch. A reader can open the markdown file while the run
    is still going.
    """
    # Update the record first. The checkpoint stores that same dict, so the
    # epoch count inside the file must already include this finished epoch.
    epoch_path = run_dir / f"epoch_{epoch:03d}.pt"
    last_path = run_dir / "model_last.pt"
    record["checkpoint"] = str(last_path)
    record["updated_at"] = _now()
    record["epochs_completed"] = epoch
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "mean": mean,
        "std": std,
        "identity": identity,
        "record": record,
    }
    torch.save(checkpoint, epoch_path)
    torch.save(checkpoint, last_path)
    _write_json(run_dir / "metrics.json", record)
    _write_json(report_json, record)
    _render_report(report_md, record)
    _render_report(run_dir / "report.md", record)


def _build_record(
    args: argparse.Namespace,
    *,
    command: str,
    smoke: bool,
    device: torch.device,
    root: Path,
    train_names: list[str],
    val_names: list[str],
    weights: list[float],
    positives: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    n_parameters: int,
    run_dir: Path,
) -> dict[str, object]:
    return {
        "model": "frame_gru",
        "framework": "pytorch",
        "command": command,
        "started_at": _now(),
        "updated_at": _now(),
        "interrupted": False,
        "smoke": smoke,
        "planned_settings": _planned_settings(args) and not smoke,
        "official_comparison": (not smoke) and _planned_settings(args),
        "device": str(device),
        "project_root": str(root),
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "epochs_completed": 0,
        "hidden": args.hidden,
        "layers": LAYERS,
        "bidirectional": BIDIRECTIONAL,
        "hidden_concat": args.hidden * 2,
        "n_parameters": n_parameters,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "batch_clips": args.batch_clips,
        "optimizer": "Adam",
        "loss": "BCEWithLogitsLoss",
        "pos_weight_formula": "n_negative_frames / n_positive_frames, or 1 when a class has no positive or no negative frames",
        "pos_weight": weights,
        "positive_frames": {name: int(count) for name, count in zip(CLASSES, positives, strict=True)},
        "threshold": THRESHOLD,
        "median_frames": MEDIAN_FRAMES,
        "threshold_tuned_on_validation": False,
        "designated_model": "last_finished_epoch",
        "feature_version": FEATURE_VERSION,
        "sample_rate": SAMPLE_RATE,
        "n_fft": N_FFT,
        "hop": HOP,
        "n_mels": N_MELS,
        "n_frames": N_FRAMES,
        "classes": list(CLASSES),
        "train_clips": len(train_names),
        "train_first": train_names[0],
        "train_last": train_names[-1],
        "val_clips": len(val_names),
        "val_first": val_names[0],
        "val_last": val_names[-1],
        "mean": mean.astype(float).tolist(),
        "std": std.astype(float).tolist(),
        "torch_version": version("torch"),
        "numpy_version": version("numpy"),
        "checkpoint": str(run_dir / "model_last.pt"),
        "log_path": str(run_dir / "train.log"),
        "logistic_baseline_reference": LOGISTIC_REFERENCE,
        "context_mlp_reference": CONTEXT_MLP_REFERENCE,
        "epochs": [],
        "first_name": val_names[0],
        "first_reference_text": "No epoch finished.",
        "first_prediction_text": "No epoch finished.",
        "closed_sets_not_read": ["public eval", "dcase2019", "real DESED lists", "FSD50K"],
    }


def run(argv: list[str] | None = None) -> Path:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, OSError):
        pass
    parser = argparse.ArgumentParser(
        description="Train a bidirectional frame GRU on synthetic DESED clips. The saved model is the last finished epoch."
    )
    parser.add_argument("--limit", type=int, default=None, help="Use the first N training clips. Omit for all 10000.")
    parser.add_argument(
        "--val-limit",
        type=int,
        default=None,
        help="Score the first N validation clips. Defaults to --limit when that is set, otherwise all 2500.",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Default 30. Resume with a higher number to continue.")
    parser.add_argument("--batch-clips", type=int, default=DEFAULT_BATCH_CLIPS, help="Clips per step. Each clip is 497 frames.")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument(
        "--hidden",
        type=int,
        default=HIDDEN_UNITS,
        help="Hidden size in each direction. 64 is the planned run. Another size uses its own folder.",
    )
    parser.add_argument("--grad-clip", type=float, default=DEFAULT_GRAD_CLIP, help="Max global gradient norm. Default 5.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=-1, help="Worker processes for feature extraction. -1 uses every CPU. Ignored when the cache is finished.")
    parser.add_argument("--device", default="cpu", help="PyTorch device. This project is set up for cpu.")
    parser.add_argument("--resume", action="store_true", help="Continue model_last.pt in this run's folder through --epochs.")
    args = parser.parse_args(argv)
    if (
        args.epochs < 1
        or args.batch_clips < 1
        or args.hidden < 1
        or args.lr <= 0
        or args.weight_decay < 0
        or args.grad_clip <= 0
    ):
        raise SystemExit("epochs, batch size, hidden size, learning rate, and gradient clip must be positive, and weight decay must be >= 0")

    root = project_root()
    jobs = resolve_jobs(args.jobs)
    # When --limit is set and --val-limit is not, the validation prefix matches
    # the training prefix. The full command omits both, so validation stays at
    # all 2500 clips. Passing --limit 10000 would ask validation for 10000
    # clips and fail, because that split has 2500. Omit --limit for the full run.
    val_limit = args.limit if args.val_limit is None else args.val_limit
    train_audio, train_names, train_labels = select_clips("train", args.limit)
    val_audio, val_names, val_labels = select_clips("validation", val_limit)
    if train_audio.resolve() == val_audio.resolve():
        raise RuntimeError("train and validation audio directories are the same")
    smoke = len(train_names) != EXPECTED_CLIPS["train"] or len(val_names) != EXPECTED_CLIPS["validation"]
    run_dir = _run_dir(smoke, len(train_names), len(val_names), args)
    run_dir.mkdir(parents=True, exist_ok=True)
    log = RunLog(run_dir / "train.log")
    report_md, report_json = _report_paths(smoke, args)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    command_bits = ["uv run python -m nndl_project.train_frame_gru"]
    if args.limit is not None:
        command_bits.append(f"--limit {args.limit}")
    if args.val_limit is not None:
        command_bits.append(f"--val-limit {args.val_limit}")
    if args.epochs != DEFAULT_EPOCHS:
        command_bits.append(f"--epochs {args.epochs}")
    if args.batch_clips != DEFAULT_BATCH_CLIPS:
        command_bits.append(f"--batch-clips {args.batch_clips}")
    if args.lr != DEFAULT_LR:
        command_bits.append(f"--lr {args.lr}")
    if args.weight_decay != DEFAULT_WEIGHT_DECAY:
        command_bits.append(f"--weight-decay {args.weight_decay}")
    if args.hidden != HIDDEN_UNITS:
        command_bits.append(f"--hidden {args.hidden}")
    if args.grad_clip != DEFAULT_GRAD_CLIP:
        command_bits.append(f"--grad-clip {args.grad_clip}")
    if args.seed != 0:
        command_bits.append(f"--seed {args.seed}")
    if args.device != "cpu":
        command_bits.append(f"--device {args.device}")
    if args.resume:
        command_bits.append("--resume")
    command = " ".join(command_bits)
    identity = _identity(args, len(train_names), len(val_names))

    log.line(f"frame GRU start {_now()}")
    log.line(f"command: {command}")
    log.line(f"project root: {root}")
    log.line(f"device: {device}")
    log.line(f"run folder: {run_dir}")
    log.line(f"report: {report_md}")
    log.line(f"machine-readable record: {report_json}")
    log.line(
        f"train {len(train_names)} clips ({train_names[0]} to {train_names[-1]}), "
        f"validation {len(val_names)} clips ({val_names[0]} to {val_names[-1]})"
    )
    log.line(
        f"architecture: bidirectional GRU, {LAYERS} layer, "
        f"input {N_MELS}, hidden {args.hidden} each direction, "
        f"linear {args.hidden * 2} to {len(CLASSES)} logits"
    )
    log.line(
        "The time axis is the 497 frames of one clip. Context does not cross clips. "
        "Loss is binary cross-entropy on logits. Decode stays median 11 then threshold 0.5."
    )
    if smoke:
        log.line("This is a prefix check. Do not quote it as the full validation score.")
    elif not _planned_settings(args):
        log.line(
            "These settings differ from the planned comparison. "
            f"Checkpoints stay in {run_dir} and the report is {report_md}."
        )
    else:
        log.line(
            "Full planned split. Logistic baseline: "
            f"segment micro {LOGISTIC_REFERENCE['segment_micro_f1']}, "
            f"segment macro {LOGISTIC_REFERENCE['segment_macro_f1']}, "
            f"event micro {LOGISTIC_REFERENCE['event_micro_f1']}, "
            f"event macro {LOGISTIC_REFERENCE['event_macro_f1']}."
        )
        log.line(
            "Context MLP epoch 432: "
            f"segment micro {CONTEXT_MLP_REFERENCE['segment_micro_f1']}, "
            f"segment macro {CONTEXT_MLP_REFERENCE['segment_macro_f1']}, "
            f"event micro {CONTEXT_MLP_REFERENCE['event_micro_f1']}, "
            f"event macro {CONTEXT_MLP_REFERENCE['event_macro_f1']}."
        )
    log.line("Ctrl+C keeps the last epoch that finished and drops the epoch in progress. Exit code 130.")
    log.line("The saved model is that last finished epoch, not a best-validation pick. The threshold stays at 0.5.")
    log.line("Public eval and dcase2019 are not read.")
    log.line(
        "On CPU, read the eta of the first epoch. If it is more than about 15 minutes, "
        "stop and rerun with a smaller --hidden. That size uses a different folder."
    )

    # A finished cache is reused. The 10000 and 2500 caches from the baseline
    # already match feature version 1, so a full run should not rebuild them.
    log.line("loading feature caches (a finished cache is reused, not rebuilt)")
    train_features, train_targets = build_feature_cache("train", train_audio, train_names, train_labels, jobs)
    val_features, val_targets = build_feature_cache("validation", val_audio, val_names, val_labels, jobs)
    if train_features.shape[1:] != (N_FRAMES, N_MELS) or val_features.shape[1:] != (N_FRAMES, N_MELS):
        raise RuntimeError(
            f"cache layout is train {tuple(train_features.shape)} validation {tuple(val_features.shape)}, "
            f"expected (*, {N_FRAMES}, {N_MELS})"
        )
    mean, std = feature_moments(train_features)
    positives = positive_frame_counts(train_targets)
    n_train_frames = len(train_names) * N_FRAMES
    weights = _pos_weight(positives, n_train_frames)
    for name, n_pos, weight in zip(CLASSES, positives, weights, strict=True):
        log.line(
            f"class {name}: positive frames {int(n_pos)} "
            f"({100.0 * float(n_pos) / n_train_frames:.2f} percent), positive weight {weight:.2f}"
        )

    model = FrameGRU(n_mels=N_MELS, hidden=args.hidden, n_classes=len(CLASSES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # pos_weight shape (10,) matches the class axis, which is the last axis of
    # both the logits and the labels. It broadcasts over the batch and the frames.
    pos_weight = torch.tensor(weights, dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="mean")
    # The unweighted loss is logged only. It is not the quantity Adam steps on.
    # It shows whether the weighted loss is being driven by the rare classes.
    unweighted = nn.BCEWithLogitsLoss(reduction="mean")
    n_parameters = sum(parameter.numel() for parameter in model.parameters())
    log.line(f"parameters: {n_parameters}")

    record = _build_record(
        args,
        command=command,
        smoke=smoke,
        device=device,
        root=root,
        train_names=train_names,
        val_names=val_names,
        weights=weights,
        positives=positives,
        mean=mean,
        std=std,
        n_parameters=n_parameters,
        run_dir=run_dir,
    )
    start_epoch = 0
    if args.resume:
        payload = _load_resume(run_dir / "model_last.pt", identity, log)
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        # Keep the standardization that was used to train the checkpoint, even
        # though a fresh computation from the same cache should match it.
        mean = np.asarray(payload["mean"], dtype=np.float32)
        std = np.asarray(payload["std"], dtype=np.float32)
        saved_record = payload["record"]
        assert isinstance(saved_record, dict)
        saved_weights = saved_record.get("pos_weight")
        if isinstance(saved_weights, list) and len(saved_weights) == len(weights):
            if any(abs(float(old) - new) > 1e-4 for old, new in zip(saved_weights, weights, strict=True)):
                raise RuntimeError(
                    "Positive class weights no longer match the checkpoint. "
                    "The feature cache may have changed. Refusing to resume."
                )
        record["epochs"] = saved_record.get("epochs", [])
        record["started_at"] = saved_record.get("started_at", record["started_at"])
        record["first_reference_text"] = saved_record.get("first_reference_text", record["first_reference_text"])
        record["first_prediction_text"] = saved_record.get("first_prediction_text", record["first_prediction_text"])
        start_epoch = int(payload["epoch"])
        record["epochs_completed"] = start_epoch
        # epochs_requested on the new record is the new --epochs. The old
        # request is not copied, so the report shows how far this command asked to go.
        log.line(f"loaded weights, Adam state, and standardization from epoch {start_epoch}")
    elif (run_dir / "model_last.pt").is_file():
        log.line(
            f"A previous checkpoint is in {run_dir}. This new run replaces it as epochs finish. "
            "Use --resume to continue it."
        )

    if start_epoch >= args.epochs:
        log.line(f"Checkpoint already has epoch {start_epoch}, and --epochs is {args.epochs}. Nothing to do.")
        log.close()
        return report_md

    try:
        for epoch in range(start_epoch + 1, args.epochs + 1):
            epoch_started = time.perf_counter()
            # seed + epoch, not a fresh unseeded generator. A resumed run then
            # shuffles epoch N the same way it did the first time.
            torch.manual_seed(args.seed + epoch)
            epoch_generator = np.random.default_rng(args.seed + epoch)
            log.line(f"epoch {epoch}/{args.epochs} training")
            _unused_probs, train_weighted, train_plain = _forward_epoch(
                model,
                train_features,
                train_targets,
                mean,
                std,
                train=True,
                optimizer=optimizer,
                criterion=criterion,
                unweighted=unweighted,
                batch_clips=args.batch_clips,
                grad_clip=args.grad_clip,
                device=device,
                log=log,
                epoch_label=f"epoch {epoch}/{args.epochs} train",
                generator=epoch_generator,
            )
            log.line(f"epoch {epoch}/{args.epochs} scoring validation")
            probabilities, val_weighted, val_plain = _forward_epoch(
                model,
                val_features,
                val_targets,
                mean,
                std,
                train=False,
                optimizer=None,
                criterion=criterion,
                unweighted=unweighted,
                batch_clips=args.batch_clips,
                grad_clip=args.grad_clip,
                device=device,
                log=log,
                epoch_label=f"epoch {epoch}/{args.epochs} val",
                generator=np.random.default_rng(0),
            )
            assert probabilities is not None
            # score_split applies median filter 11 and threshold 0.5. Do not
            # pass another threshold here. Tuning it on this split would spend
            # the validation set.
            segment_totals, event_totals, first_reference, first_prediction = score_split(
                val_names, val_labels, probabilities
            )
            elapsed = time.perf_counter() - epoch_started
            segment_payload = _counts_payload(segment_totals)
            event_payload = _counts_payload(event_totals)
            row = {
                "epoch": epoch,
                "seconds": elapsed,
                "learning_rate": args.lr,
                "train_loss_weighted": train_weighted,
                "train_loss_unweighted": train_plain,
                "val_loss_weighted": val_weighted,
                "val_loss_unweighted": val_plain,
                "segment": segment_payload,
                "event": event_payload,
            }
            epochs_so_far = record["epochs"]
            assert isinstance(epochs_so_far, list)
            epochs_so_far.append(row)
            record["first_reference_text"] = _event_lines(first_reference)
            record["first_prediction_text"] = _event_lines(first_prediction)
            _save_epoch(record, model, optimizer, mean, std, identity, run_dir, report_md, report_json, epoch)
            segment_micro = segment_payload["micro"]
            event_micro = event_payload["micro"]
            assert isinstance(segment_micro, dict) and isinstance(event_micro, dict)
            log.line(
                f"epoch {epoch}/{args.epochs} done in {elapsed:.1f}s "
                f"train weighted {train_weighted:.4f} unweighted {train_plain:.4f} "
                f"val weighted {val_weighted:.4f} unweighted {val_plain:.4f}"
            )
            log.line(
                f"segment micro F1 {_fmt(segment_micro['f1'])} macro {_fmt(segment_payload['macro_f1'])} | "
                f"event micro F1 {_fmt(event_micro['f1'])} macro {_fmt(event_payload['macro_f1'])}"
            )
            classes = segment_payload["classes"]
            event_classes = event_payload["classes"]
            assert isinstance(classes, dict) and isinstance(event_classes, dict)
            for name in CLASSES:
                segment_row = classes[name]
                event_row = event_classes[name]
                assert isinstance(segment_row, dict) and isinstance(event_row, dict)
                log.line(
                    f"  {name}: segment P {_fmt(segment_row['precision'])} R {_fmt(segment_row['recall'])} "
                    f"F1 {_fmt(segment_row['f1'])} tp {segment_row['tp']} fp {segment_row['fp']} fn {segment_row['fn']} | "
                    f"event F1 {_fmt(event_row['f1'])} tp {event_row['tp']} fp {event_row['fp']} fn {event_row['fn']}"
                )
            log.line(f"saved {run_dir / f'epoch_{epoch:03d}.pt'}")
            log.line(f"saved {run_dir / 'model_last.pt'}")
            log.line(f"saved {report_md}")
    except KeyboardInterrupt:
        record["interrupted"] = True
        record["updated_at"] = _now()
        _write_json(run_dir / "metrics.json", record)
        _write_json(report_json, record)
        _render_report(report_md, record)
        finished = record["epochs_completed"]
        log.line("Interrupted by Ctrl+C.")
        log.line("The epoch that was in progress was not saved.")
        if finished:
            log.line(f"Last finished epoch is {finished}. Checkpoint: {run_dir / 'model_last.pt'}")
            log.line(f"Continue with the same settings plus --resume, and --epochs greater than {finished}.")
        else:
            log.line("No epoch finished, so there is no checkpoint yet.")
        log.close()
        raise SystemExit(130) from None
    except Exception:
        log.line("The run failed:")
        log.line(traceback.format_exc())
        record["interrupted"] = True
        record["error"] = traceback.format_exc()
        record["updated_at"] = _now()
        _write_json(run_dir / "metrics.json", record)
        _write_json(report_json, record)
        _render_report(report_md, record)
        log.close()
        raise

    log.line(f"DONE {_now()}: wrote {report_md}")
    log.line("The designated model is the last finished epoch. Do not pick an earlier epoch from the table as the official score.")
    log.close()
    return report_md


def main() -> None:
    run()


if __name__ == "__main__":
    main()
