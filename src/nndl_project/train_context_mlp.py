"""Train the context multilayer perceptron and save every finished epoch.

Run this yourself from the project directory, in a terminal you can watch:

    uv run python -m nndl_project.train_context_mlp --limit 200
    uv run python -m nndl_project.train_context_mlp

The first command is a 200-clip pipeline check. The second trains on all
10000 synthetic21_train clips and scores all 2500 synthetic21_validation
clips. Ctrl+C stops the run. The last epoch that finished is kept.

The designated model is the last finished epoch. Validation scores are printed
so you can watch the run. They are not used to pick a better epoch or to
move the 0.5 threshold.

Public eval and the dcase2019 folders are not read.
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
from nndl_project.context import CONTEXT_RADIUS, HIDDEN_UNITS, ContextMLP, context_frames, n_inputs, stack_context
from nndl_project.data import Event, project_root
from nndl_project.metrics import Counts, macro_f1, micro_counts, precision_recall_f1

DEFAULT_EPOCHS = 10
DEFAULT_BATCH_CLIPS = 16
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 0.0
LOGISTIC_REFERENCE = {
    "segment_micro_f1": 0.393,
    "segment_macro_f1": 0.289,
    "event_micro_f1": 0.044,
    "event_macro_f1": 0.030,
    "split": "all 2500 synthetic21_validation clips",
    "note": "Frame-wise logistic regression with threshold 0.5. Compare a context MLP run with these numbers only when that run also scores all 2500 validation clips.",
}


class RunLog:
    """Print each line immediately and append the same line to a log file."""

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
            f"| {name} | {_fmt(row['precision'])} | {_fmt(row['recall'])} | {_fmt(row['f1'])} | {row['tp']} | {row['fp']} | {row['fn']} |"
        )
    micro = payload["micro"]
    assert isinstance(micro, dict)
    lines.append(
        f"| micro | {_fmt(micro['precision'])} | {_fmt(micro['recall'])} | {_fmt(micro['f1'])} | {micro['tp']} | {micro['fp']} | {micro['fn']} |"
    )
    macro = payload["macro_f1"]
    lines.append(f"| macro f1 |  |  | {_fmt(macro)} |  |  |  |")
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
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _batch_windows(
    raw: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    radius: int,
) -> np.ndarray:
    """Normalize a clip batch and flatten each frame's context window."""
    normalized = (raw - mean) / std
    windows = np.empty((raw.shape[0], raw.shape[1], n_inputs(radius)), dtype=np.float32)
    for index in range(raw.shape[0]):
        windows[index] = stack_context(normalized[index], radius)
    return windows


def _pos_weight(positives: np.ndarray, n_frames: int) -> list[float]:
    weights: list[float] = []
    for n_pos in positives:
        n_neg = n_frames - int(n_pos)
        if int(n_pos) <= 0 or n_neg <= 0:
            weights.append(1.0)
        else:
            weights.append(n_neg / float(n_pos))
    return weights


def _run_dir(smoke: bool, train_count: int, val_count: int) -> Path:
    name = f"limit{train_count}_val{val_count}" if smoke else "full"
    return project_root() / "data" / "cache" / "context_mlp" / name


def _report_paths(smoke: bool) -> tuple[Path, Path]:
    reports = project_root() / "reports"
    if smoke:
        return reports / "context_mlp_smoke.md", reports / "context_mlp_smoke.json"
    return reports / "context_mlp.md", reports / "context_mlp.json"


def _identity(args: argparse.Namespace, train_count: int, val_count: int) -> dict[str, object]:
    return {
        "context_radius": args.radius,
        "hidden": args.hidden,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "batch_clips": args.batch_clips,
        "seed": args.seed,
        "train_clips": train_count,
        "val_clips": val_count,
    }


def _load_resume(path: Path, identity: dict[str, object], log: RunLog) -> dict[str, object] | None:
    if not path.is_file():
        raise FileNotFoundError(f"--resume was set, but there is no checkpoint at {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    saved = payload.get("identity")
    if saved != identity:
        raise RuntimeError(
            "The checkpoint was trained with different settings. "
            f"Saved {saved}. This command is {identity}. Start a new run, or repeat the same settings."
        )
    log.line(f"resuming from epoch {payload['epoch']} in {path}")
    return payload


def _forward_epoch(
    model: ContextMLP,
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
    radius: int,
    device: torch.device,
    log: RunLog,
    epoch_label: str,
    generator: np.random.Generator,
) -> tuple[np.ndarray | None, float, float]:
    n_clips = features.shape[0]
    if train:
        assert optimizer is not None
        model.train()
        order = generator.permutation(n_clips)
    else:
        model.eval()
        order = np.arange(n_clips)
        probabilities = np.empty((n_clips, N_FRAMES, len(CLASSES)), dtype=np.float32)
    weighted_sum = 0.0
    unweighted_sum = 0.0
    seen = 0
    started = time.perf_counter()
    n_batches = int(np.ceil(n_clips / batch_clips))
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch_index, start in enumerate(range(0, n_clips, batch_clips), start=1):
            batch_ids = order[start : start + batch_clips]
            raw = np.asarray(features[batch_ids], dtype=np.float32)
            windows = _batch_windows(raw, mean, std, radius)
            design = torch.from_numpy(windows.reshape(-1, windows.shape[-1])).to(device)
            labels_np = np.asarray(targets[batch_ids], dtype=np.float32).reshape(-1, len(CLASSES))
            labels = torch.from_numpy(labels_np).to(device)
            logits = model(design)
            weighted_loss = criterion(logits, labels)
            plain_loss = unweighted(logits, labels)
            if train:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                weighted_loss.backward()
                optimizer.step()
            else:
                probabilities[batch_ids] = (
                    torch.sigmoid(logits).detach().cpu().numpy().reshape(raw.shape[0], N_FRAMES, len(CLASSES))
                )
            elements = int(labels.numel())
            weighted_sum += float(weighted_loss.item()) * elements
            unweighted_sum += float(plain_loss.item()) * elements
            seen += elements
            elapsed = time.perf_counter() - started
            clips_done = start + raw.shape[0]
            rate = clips_done / elapsed if elapsed > 0 else 0.0
            remaining = (n_clips - clips_done) / rate if rate > 0 else 0.0
            log.line(
                f"{epoch_label} batch {batch_index}/{n_batches} "
                f"clips {clips_done}/{n_clips} "
                f"weighted loss {weighted_loss.item():.4f} "
                f"epoch avg {weighted_sum / seen:.4f} "
                f"unweighted avg {unweighted_sum / seen:.4f} "
                f"{rate:.0f} clips/s eta {remaining / 60:.1f} min"
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
            "It is not the full-data context MLP score."
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
            "These scores are the context multilayer perceptron on the clips named below. "
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
    reference = record["first_reference_text"]
    prediction = record["first_prediction_text"]
    text = f"""# Context multilayer perceptron

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
- Context: {record["context_frames"]} frames ({record["context_radius"]} on each side), {record["n_inputs"]} inputs.
- Network: linear {record["n_inputs"]} to {record["hidden"]}, rectified linear unit, linear {record["hidden"]} to 10, sigmoid per class.
- Loss: binary cross-entropy on logits. Positive weight for a class is negative frames divided by positive frames: {weight_text}.
- Optimizer: Adam, learning rate {record["learning_rate"]}, weight decay {record["weight_decay"]}, batch {record["batch_clips"]} clips, seed {record["seed"]}.
- Decode: median filter of {record["median_frames"]} frames, threshold {record["threshold"]}.
- Feature cache version {record["feature_version"]}: {record["n_mels"]} mel bands, FFT {record["n_fft"]}, hop {record["hop"]} at {record["sample_rate"]} Hz, {record["n_frames"]} frames.
- PyTorch {record["torch_version"]}, NumPy {record["numpy_version"]}.
- Checkpoint: `{record["checkpoint"]}`.
- Log: `{record["log_path"]}`.
- Epochs requested: {record["epochs_requested"]}. Epochs finished: {record["epochs_completed"]}.

Logistic baseline on all 2500 validation clips, for comparison when this run is also the full split: segment micro F1 {LOGISTIC_REFERENCE["segment_micro_f1"]}, segment macro F1 {LOGISTIC_REFERENCE["segment_macro_f1"]}, event micro F1 {LOGISTIC_REFERENCE["event_micro_f1"]}, event macro F1 {LOGISTIC_REFERENCE["event_macro_f1"]}.

## First validation clip

Clip: {record["first_name"]}

Reference labels:

{reference}

Model events from the last finished epoch:

{prediction}

## Epoch history

{chr(10).join(history)}

## Last finished epoch

{last_tables}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _save_epoch(
    record: dict[str, object],
    model: ContextMLP,
    optimizer: torch.optim.Optimizer,
    mean: np.ndarray,
    std: np.ndarray,
    identity: dict[str, object],
    run_dir: Path,
    report_md: Path,
    report_json: Path,
    epoch: int,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "mean": mean,
        "std": std,
        "identity": identity,
        "record": record,
    }
    epoch_path = run_dir / f"epoch_{epoch:03d}.pt"
    last_path = run_dir / "model_last.pt"
    torch.save(checkpoint, epoch_path)
    torch.save(checkpoint, last_path)
    record["checkpoint"] = str(last_path)
    record["updated_at"] = _now()
    record["epochs_completed"] = epoch
    _write_json(run_dir / "metrics.json", record)
    _write_json(report_json, record)
    _render_report(report_md, record)
    _render_report(run_dir / "report.md", record)


def run(argv: list[str] | None = None) -> Path:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, OSError):
        pass
    parser = argparse.ArgumentParser(description="Train the context multilayer perceptron on synthetic DESED clips.")
    parser.add_argument("--limit", type=int, default=None, help="Use the first N training clips. Omit for all 10000.")
    parser.add_argument("--val-limit", type=int, default=None, help="Score the first N validation clips. Defaults to --limit when that is set.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-clips", type=int, default=DEFAULT_BATCH_CLIPS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--hidden", type=int, default=HIDDEN_UNITS)
    parser.add_argument("--radius", type=int, default=CONTEXT_RADIUS, help="Frames on each side of the frame being classified.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=-1, help="Worker processes for feature extraction. -1 uses every CPU.")
    parser.add_argument("--device", default="cpu", help="PyTorch device. This project is set up for cpu.")
    parser.add_argument("--resume", action="store_true", help="Continue the checkpoint in this run's folder through --epochs.")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_clips < 1 or args.radius < 0 or args.hidden < 1 or args.lr <= 0:
        raise SystemExit("epochs, batch size, hidden size, and learning rate must be positive, and radius must be >= 0")

    root = project_root()
    jobs = resolve_jobs(args.jobs)
    val_limit = args.limit if args.val_limit is None else args.val_limit
    train_audio, train_names, train_labels = select_clips("train", args.limit)
    val_audio, val_names, val_labels = select_clips("validation", val_limit)
    if train_audio.resolve() == val_audio.resolve():
        raise RuntimeError("train and validation audio directories are the same")
    smoke = len(train_names) != EXPECTED_CLIPS["train"] or len(val_names) != EXPECTED_CLIPS["validation"]
    run_dir = _run_dir(smoke, len(train_names), len(val_names))
    run_dir.mkdir(parents=True, exist_ok=True)
    log = RunLog(run_dir / "train.log")
    report_md, report_json = _report_paths(smoke)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    command_bits = ["uv run python -m nndl_project.train_context_mlp"]
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
    if args.radius != CONTEXT_RADIUS:
        command_bits.append(f"--radius {args.radius}")
    if args.seed != 0:
        command_bits.append(f"--seed {args.seed}")
    if args.device != "cpu":
        command_bits.append(f"--device {args.device}")
    if args.resume:
        command_bits.append("--resume")
    command = " ".join(command_bits)
    identity = _identity(args, len(train_names), len(val_names))

    log.line(f"context MLP start {_now()}")
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
    if smoke:
        log.line("This is a prefix check. Do not quote it as the full validation score.")
    else:
        log.line(
            "Full split. Logistic baseline to beat on this same validation set: "
            f"segment micro {LOGISTIC_REFERENCE['segment_micro_f1']}, "
            f"segment macro {LOGISTIC_REFERENCE['segment_macro_f1']}, "
            f"event micro {LOGISTIC_REFERENCE['event_micro_f1']}, "
            f"event macro {LOGISTIC_REFERENCE['event_macro_f1']}."
        )
    log.line("Ctrl+C keeps the last epoch that finished. The saved model is that epoch, not a best-validation pick.")
    log.line("Public eval and dcase2019 are not read.")

    train_features, train_targets = build_feature_cache("train", train_audio, train_names, train_labels, jobs)
    val_features, val_targets = build_feature_cache("validation", val_audio, val_names, val_labels, jobs)
    mean, std = feature_moments(train_features)
    positives = positive_frame_counts(train_targets)
    n_train_frames = len(train_names) * N_FRAMES
    weights = _pos_weight(positives, n_train_frames)
    for name, n_pos, weight in zip(CLASSES, positives, weights, strict=True):
        log.line(
            f"class {name}: positive frames {int(n_pos)} "
            f"({100.0 * float(n_pos) / n_train_frames:.2f} percent), positive weight {weight:.2f}"
        )

    model = ContextMLP(n_inputs(args.radius), args.hidden, len(CLASSES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    pos_weight = torch.tensor(weights, dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="mean")
    unweighted = nn.BCEWithLogitsLoss(reduction="mean")
    n_parameters = sum(parameter.numel() for parameter in model.parameters())
    log.line(f"parameters: {n_parameters}")

    record: dict[str, object] = {
        "model": "context_mlp",
        "framework": "pytorch",
        "command": command,
        "started_at": _now(),
        "updated_at": _now(),
        "interrupted": False,
        "smoke": smoke,
        "device": str(device),
        "project_root": str(root),
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "epochs_completed": 0,
        "context_radius": args.radius,
        "context_frames": context_frames(args.radius),
        "hidden": args.hidden,
        "n_inputs": n_inputs(args.radius),
        "n_parameters": n_parameters,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "batch_clips": args.batch_clips,
        "optimizer": "Adam",
        "loss": "BCEWithLogitsLoss",
        "pos_weight_formula": "n_negative_frames / n_positive_frames, or 1 when a class has no positive or no negative frames",
        "pos_weight": weights,
        "positive_frames": {name: int(count) for name, count in zip(CLASSES, positives, strict=True)},
        "threshold": THRESHOLD,
        "median_frames": MEDIAN_FRAMES,
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
        "epochs": [],
        "first_name": val_names[0],
        "first_reference_text": "No epoch finished.",
        "first_prediction_text": "No epoch finished.",
    }
    start_epoch = 0
    if args.resume:
        payload = _load_resume(run_dir / "model_last.pt", identity, log)
        assert payload is not None
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        mean = np.asarray(payload["mean"], dtype=np.float32)
        std = np.asarray(payload["std"], dtype=np.float32)
        saved_record = payload["record"]
        assert isinstance(saved_record, dict)
        record["epochs"] = saved_record.get("epochs", [])
        record["started_at"] = saved_record.get("started_at", record["started_at"])
        start_epoch = int(payload["epoch"])
        record["epochs_completed"] = start_epoch
    elif (run_dir / "model_last.pt").is_file():
        log.line(f"A previous checkpoint is in {run_dir}. This new run replaces it as epochs finish. Use --resume to continue it.")

    if start_epoch >= args.epochs:
        log.line(f"Checkpoint already has epoch {start_epoch}, and --epochs is {args.epochs}. Nothing to do.")
        log.close()
        return report_md

    try:
        for epoch in range(start_epoch + 1, args.epochs + 1):
            epoch_started = time.perf_counter()
            # The shuffle uses the process seed plus the epoch so a resumed run stays repeatable.
            torch.manual_seed(args.seed + epoch)
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
                radius=args.radius,
                device=device,
                log=log,
                epoch_label=f"epoch {epoch}/{args.epochs} train",
                generator=np.random.default_rng(args.seed + epoch),
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
                radius=args.radius,
                device=device,
                log=log,
                epoch_label=f"epoch {epoch}/{args.epochs} val",
                generator=np.random.default_rng(0),
            )
            assert probabilities is not None
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
            log.line(f"saved {report_md}")
    except KeyboardInterrupt:
        record["interrupted"] = True
        record["updated_at"] = _now()
        _write_json(run_dir / "metrics.json", record)
        _write_json(report_json, record)
        _render_report(report_md, record)
        finished = record["epochs_completed"]
        log.line("Interrupted by Ctrl+C.")
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
    log.close()
    return report_md


def main() -> None:
    run()


if __name__ == "__main__":
    main()
