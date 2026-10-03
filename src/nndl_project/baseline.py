"""Frame-wise logistic regression on synthetic DESED clips.

Smoke check, 200 training clips, validation prefix, not the reported score:

    uv run python -m nndl_project.baseline --limit 200

Full fit on all 10,000 synthetic21_train clips, scored on all
synthetic21_validation clips:

    uv run python -m nndl_project.baseline

Public eval and the dcase2019 folders are not read. A limited run writes
reports/baseline_logistic_smoke.md and does not overwrite the full-data note.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import SGDClassifier

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
from nndl_project.data import (
    Event,
    events_for_split,
    list_clip_names,
    project_root,
    split_paths,
)
from nndl_project.decode import decode_events
from nndl_project.features import extract_named_clip
from nndl_project.metrics import (
    Counts,
    accumulate,
    empty_counts,
    event_counts_for_clip,
    macro_f1,
    micro_counts,
    precision_recall_f1,
    segment_counts_for_clip,
)

SGD_ALPHA = 1e-4
DEFAULT_EPOCHS = 5
DEFAULT_BATCH_CLIPS = 32
EXPECTED_CLIPS = {"train": 10_000, "validation": 2_500}


def resolve_jobs(jobs: int) -> int:
    if jobs == 0 or jobs < -1:
        raise ValueError("--jobs must be -1 or a positive integer")
    if jobs == -1:
        return max(1, os.cpu_count() or 1)
    return jobs


def select_clips(split: str, limit: int | None) -> tuple[Path, list[str], dict[str, list[Event]]]:
    audio_dir, _tsv = split_paths(split)
    if not audio_dir.is_dir():
        raise FileNotFoundError(f"missing {split} audio directory: {audio_dir}")
    names = list_clip_names(audio_dir)
    expected = EXPECTED_CLIPS[split]
    if len(names) != expected:
        raise RuntimeError(f"{split} has {len(names)} wavs, expected {expected}. The extract looks incomplete.")
    labels = events_for_split(split)
    missing = sorted(set(labels) - set(names))
    if missing:
        raise FileNotFoundError(f"{split} labels name {len(missing)} wavs that are not in the folder, first {missing[0]}")
    if limit is None:
        chosen = names
    else:
        if limit < 1 or limit > len(names):
            raise ValueError(f"--limit / --val-limit must be from 1 to {len(names)} for {split}")
        chosen = names[:limit]
    return audio_dir, chosen, labels


def _cache_dir(split: str, n_clips: int) -> Path:
    return project_root() / "data" / "cache" / f"logmel_v{FEATURE_VERSION}" / f"{split}_{n_clips}"


def _feature_meta(split: str, names: list[str], completed: int) -> dict[str, object]:
    return {
        "feature_version": FEATURE_VERSION,
        "split": split,
        "n_clips": len(names),
        "completed": completed,
        "first": names[0],
        "last": names[-1],
        "n_frames": N_FRAMES,
        "n_mels": N_MELS,
        "n_classes": len(CLASSES),
        "sample_rate": SAMPLE_RATE,
        "n_fft": N_FFT,
        "hop": HOP,
        "classes": list(CLASSES),
    }


def _meta_compatible(saved: dict[str, object], split: str, names: list[str]) -> bool:
    expected = _feature_meta(split, names, completed=int(saved.get("completed", 0)))
    for key, value in expected.items():
        if key == "completed":
            continue
        if saved.get(key) != value:
            return False
    return True


def _write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_feature_cache(
    split: str,
    audio_dir: Path,
    names: list[str],
    labels: dict[str, list[Event]],
    jobs: int,
) -> tuple[np.memmap, np.memmap]:
    """Cache log-mel frames and frame targets. A finished cache is reused."""
    cache_dir = _cache_dir(split, len(names))
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / "meta.json"
    feature_path = cache_dir / "features.f32"
    target_path = cache_dir / "targets.u8"
    names_path = cache_dir / "names.txt"
    feature_shape = (len(names), N_FRAMES, N_MELS)
    target_shape = (len(names), N_FRAMES, len(CLASSES))
    feature_bytes = int(np.prod(feature_shape)) * 4
    target_bytes = int(np.prod(target_shape))
    saved: dict[str, object] | None = None
    if meta_path.is_file():
        saved = json.loads(meta_path.read_text(encoding="utf-8"))
    reusable = (
        saved is not None
        and names_path.is_file()
        and _meta_compatible(saved, split, names)
        and feature_path.is_file()
        and target_path.is_file()
        and feature_path.stat().st_size == feature_bytes
        and target_path.stat().st_size == target_bytes
        and names_path.read_text(encoding="utf-8").splitlines() == names
    )
    if not reusable:
        for path in (feature_path, target_path, names_path, meta_path):
            if path.exists():
                path.unlink()
        names_path.write_text("\n".join(names) + "\n", encoding="utf-8")
        features = np.memmap(feature_path, dtype=np.float32, mode="w+", shape=feature_shape)
        targets = np.memmap(target_path, dtype=np.uint8, mode="w+", shape=target_shape)
        completed = 0
        _write_json(meta_path, _feature_meta(split, names, completed))
    else:
        assert saved is not None
        completed = int(saved["completed"])
        features = np.memmap(feature_path, dtype=np.float32, mode="r+", shape=feature_shape)
        targets = np.memmap(target_path, dtype=np.uint8, mode="r+", shape=target_shape)
        if completed >= len(names):
            print(f"reusing {split} features for {len(names)} clips", flush=True)
            return _readonly(feature_path, feature_shape, target_path, target_shape, features, targets)

    print(f"building {split} features from clip {completed} of {len(names)}", flush=True)
    chunk = max(jobs, 1) * 2
    with Parallel(n_jobs=jobs, prefer="processes") as pool:
        while completed < len(names):
            stop = min(completed + chunk, len(names))
            batch = names[completed:stop]
            extracted = pool(
                delayed(extract_named_clip)(str(audio_dir), name, labels.get(name, []))
                for name in batch
            )
            for offset, (frame_features, frame_labels) in enumerate(extracted):
                features[completed + offset] = frame_features
                targets[completed + offset] = frame_labels
            features.flush()
            targets.flush()
            completed = stop
            _write_json(meta_path, _feature_meta(split, names, completed))
            print(f"{split} features {completed}/{len(names)}", flush=True)
    return _readonly(feature_path, feature_shape, target_path, target_shape, features, targets)


def _readonly(
    feature_path: Path,
    feature_shape: tuple[int, int, int],
    target_path: Path,
    target_shape: tuple[int, int, int],
    features: np.memmap,
    targets: np.memmap,
) -> tuple[np.memmap, np.memmap]:
    features.flush()
    targets.flush()
    del features, targets
    return (
        np.memmap(feature_path, dtype=np.float32, mode="r", shape=feature_shape),
        np.memmap(target_path, dtype=np.uint8, mode="r", shape=target_shape),
    )


def feature_moments(features: np.memmap) -> tuple[np.ndarray, np.ndarray]:
    total = np.zeros(N_MELS, dtype=np.float64)
    total_sq = np.zeros(N_MELS, dtype=np.float64)
    count = 0
    for start in range(0, features.shape[0], 32):
        block = np.asarray(features[start : start + 32], dtype=np.float64)
        total += block.sum(axis=(0, 1))
        total_sq += np.square(block).sum(axis=(0, 1))
        count += block.shape[0] * block.shape[1]
    mean = total / count
    variance = np.maximum(total_sq / count - np.square(mean), 1e-12)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def positive_frame_counts(targets: np.memmap) -> np.ndarray:
    counts = np.zeros(len(CLASSES), dtype=np.int64)
    for start in range(0, targets.shape[0], 64):
        block = np.asarray(targets[start : start + 64], dtype=np.int64)
        counts += block.sum(axis=(0, 1))
    return counts


def fit_classifiers(
    features: np.memmap,
    targets: np.memmap,
    *,
    epochs: int,
    batch_clips: int,
    seed: int,
) -> tuple[list[SGDClassifier], np.ndarray, np.ndarray, np.ndarray]:
    """Fit one logistic regression per class with averaged SGD.

    max_iter is 1 because each partial_fit call is one pass over a batch.
    The epoch loop is the only place that revisits the training clips.
    Sample weights use the global positive and negative frame counts.
    """
    if epochs < 1:
        raise ValueError("--epochs must be positive")
    if batch_clips < 1:
        raise ValueError("--batch-clips must be positive")
    mean, std = feature_moments(features)
    positives = positive_frame_counts(targets)
    n_frames = int(features.shape[0] * N_FRAMES)
    weight_pos = np.ones(len(CLASSES), dtype=np.float64)
    weight_neg = np.ones(len(CLASSES), dtype=np.float64)
    for index, n_pos in enumerate(positives):
        n_neg = n_frames - int(n_pos)
        if n_pos > 0 and n_neg > 0:
            weight_pos[index] = n_frames / (2.0 * float(n_pos))
            weight_neg[index] = n_frames / (2.0 * float(n_neg))
    models = [
        SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=SGD_ALPHA,
            fit_intercept=True,
            max_iter=1,
            tol=None,
            shuffle=False,
            learning_rate="optimal",
            average=True,
            random_state=seed + index,
        )
        for index in range(len(CLASSES))
    ]
    class_ids = np.array([0, 1], dtype=np.int8)
    n_clips = features.shape[0]
    for epoch in range(epochs):
        started = time.perf_counter()
        order = np.random.default_rng(seed + epoch).permutation(n_clips)
        for start in range(0, n_clips, batch_clips):
            batch_ids = order[start : start + batch_clips]
            raw = np.asarray(features[batch_ids], dtype=np.float32)
            design = ((raw - mean) / std).reshape(-1, N_MELS)
            labels = np.asarray(targets[batch_ids], dtype=np.int8).reshape(-1, len(CLASSES))
            for index, model in enumerate(models):
                y = labels[:, index]
                weights = np.where(y == 1, weight_pos[index], weight_neg[index])
                model.partial_fit(design, y, classes=class_ids, sample_weight=weights)
        losses = _probe_log_loss(models, features, targets, mean, std)
        loss_text = ", ".join(f"{name} {loss:.3f}" for name, loss in zip(CLASSES, losses, strict=True))
        print(
            f"epoch {epoch + 1}/{epochs} in {time.perf_counter() - started:.1f}s probe log loss {loss_text}",
            flush=True,
        )
    return models, mean, std, positives


def _probe_log_loss(
    models: list[SGDClassifier],
    features: np.memmap,
    targets: np.memmap,
    mean: np.ndarray,
    std: np.ndarray,
) -> list[float]:
    """Unweighted log loss on the first 16 training clips. This is not a score."""
    n_probe = min(16, features.shape[0])
    raw = np.asarray(features[:n_probe], dtype=np.float32)
    design = ((raw - mean) / std).reshape(-1, N_MELS)
    labels = np.asarray(targets[:n_probe], dtype=np.float64).reshape(-1, len(CLASSES))
    losses: list[float] = []
    for index, model in enumerate(models):
        probability = np.clip(model.predict_proba(design)[:, 1], 1e-6, 1.0 - 1e-6)
        y = labels[:, index]
        losses.append(float(-(y * np.log(probability) + (1.0 - y) * np.log(1.0 - probability)).mean()))
    return losses


def predict_probabilities(
    models: list[SGDClassifier],
    features: np.memmap,
    mean: np.ndarray,
    std: np.ndarray,
    batch_clips: int,
) -> np.ndarray:
    probabilities = np.empty((features.shape[0], N_FRAMES, len(CLASSES)), dtype=np.float32)
    for start in range(0, features.shape[0], batch_clips):
        stop = min(start + batch_clips, features.shape[0])
        raw = np.asarray(features[start:stop], dtype=np.float32)
        design = ((raw - mean) / std).reshape(-1, N_MELS)
        for index, model in enumerate(models):
            probabilities[start:stop, :, index] = model.predict_proba(design)[:, 1].reshape(stop - start, N_FRAMES)
    return probabilities


def score_split(
    names: list[str],
    labels: dict[str, list[Event]],
    probabilities: np.ndarray,
) -> tuple[dict[str, Counts], dict[str, Counts], list[Event], list[Event]]:
    if not names:
        raise ValueError("no clips to score")
    segment_totals = empty_counts()
    event_totals = empty_counts()
    first_reference: list[Event] = []
    first_prediction: list[Event] = []
    for index, name in enumerate(names):
        reference = labels.get(name, [])
        prediction = decode_events(probabilities[index], threshold=THRESHOLD, median_frames=MEDIAN_FRAMES)
        accumulate(segment_totals, segment_counts_for_clip(reference, prediction))
        accumulate(event_totals, event_counts_for_clip(reference, prediction))
        if index == 0:
            first_reference = reference
            first_prediction = prediction
    return segment_totals, event_totals, first_reference, first_prediction


def _format_metric(counts: Counts) -> str:
    row = precision_recall_f1(counts)
    if row is None:
        return "n/a | n/a | n/a"
    precision, recall, f1 = row
    return f"{precision:.3f} | {recall:.3f} | {f1:.3f}"


def _metric_table(by_class: dict[str, Counts]) -> str:
    lines = ["| class | precision | recall | f1 | tp | fp | fn |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name in CLASSES:
        counts = by_class[name]
        lines.append(f"| {name} | {_format_metric(counts)} | {counts.tp} | {counts.fp} | {counts.fn} |")
    micro = micro_counts(by_class)
    macro = macro_f1(by_class)
    micro_text = _format_metric(micro)
    macro_text = "n/a" if macro is None else f"{macro:.3f}"
    lines.append(f"| micro | {micro_text} | {micro.tp} | {micro.fp} | {micro.fn} |")
    lines.append(f"| macro f1 |  |  | {macro_text} |  |  |  |")
    return "\n".join(lines)


def _event_lines(events: list[Event]) -> str:
    if not events:
        return "None."
    lines = []
    for event in events:
        score = "" if event.score is None else f", score {event.score:.3f}"
        lines.append(f"- {event.label} from {event.onset:.3f} s to {event.offset:.3f} s{score}")
    return "\n".join(lines)


def write_report(
    path: Path,
    *,
    command: str,
    smoke: bool,
    train_names: list[str],
    val_names: list[str],
    positives: np.ndarray,
    segment_totals: dict[str, Counts],
    event_totals: dict[str, Counts],
    first_name: str,
    first_reference: list[Event],
    first_prediction: list[Event],
    elapsed_seconds: float,
    epochs: int,
    batch_clips: int,
    seed: int,
) -> None:
    n_train_frames = len(train_names) * N_FRAMES
    positive_bits = ", ".join(
        f"{name} {int(count)} ({(100.0 * count / n_train_frames):.2f} percent)"
        for name, count in zip(CLASSES, positives, strict=True)
    )
    if smoke:
        opening = (
            "This file is a pipeline check on a prefix of the synthetic clips. "
            "It is not the full-data baseline score."
        )
    else:
        opening = (
            "These scores are the frame-wise logistic regression baseline on all "
            "synthetic21_validation clips. The threshold was not tuned on this split."
        )
    text = f"""# Frame-wise logistic regression baseline

{opening}

Command: `{command}`

The classifier is one logistic regression per DESED class. Each input row is one log-mel frame. This is not a convolutional network, and the log-mel matrix is not treated as a picture.

## Setup

- Train audio and labels: synthetic21_train only, {len(train_names)} clips, {train_names[0]} through {train_names[-1]}.
- Validation audio and labels: synthetic21_validation only, {len(val_names)} clips, {val_names[0]} through {val_names[-1]}.
- Train and validation both use names such as 0.wav. Each label was joined to the wav in its own split folder.
- Clips are 10 seconds. The loader mixes to one channel and resamples 44.1 kHz to 16 kHz. There is no soundscapes_16k copy on disk.
- Log-mel features: {N_MELS} bands, FFT length {N_FFT}, hop {HOP} samples ({HOP / SAMPLE_RATE * 1000:.0f} ms), {N_FRAMES} frames per clip.
- A frame is positive when its FFT window overlaps a strong label. The ten DESED names are kept as they are.
- Optimizer: scikit-learn SGDClassifier with log loss, L2 alpha {SGD_ALPHA}, averaged weights, {epochs} epochs, batch of {batch_clips} clips, seed {seed}.
- Class imbalance uses one positive weight and one negative weight per class, from the training frame counts. Those weights are not recomputed inside a batch.
- Decode: median filter of {MEDIAN_FRAMES} frames, then threshold {THRESHOLD}. An event score is the mean smoothed probability inside the event. Boundaries use the {HOP / SAMPLE_RATE * 1000:.0f} ms hop grid.
- Segment F1 uses {1:.0f} second bins. Event F1 matches one prediction to one reference when the onset is within 200 ms and the offset is within max(200 ms, 20 percent of the reference duration).
- Public eval and the dcase2019 condition folders were not used for training, features, or threshold choices.
- Package versions: scikit-learn {version("scikit-learn")}, scipy {version("scipy")}, numpy {version("numpy")}.
- Elapsed time: {elapsed_seconds:.1f} seconds.

Positive training frames, out of {n_train_frames}: {positive_bits}.

## Output of one clip

Each detected event has a class name, a start time in seconds, an end time in seconds, and a score from 0 to 1. Overlapping classes are separate events. The first validation clip in this run is {first_name}.

Reference labels:

{_event_lines(first_reference)}

Baseline events:

{_event_lines(first_prediction)}

## Segment scores

{_metric_table(segment_totals)}

## Event scores

{_metric_table(event_totals)}

A class row is n/a when that class has no reference events and no predictions in the scored clips. Macro F1 averages the classes that have a defined F1. Micro F1 pools tp, fp, and fn across classes.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def run(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Fit and score the frame-wise logistic regression baseline.")
    parser.add_argument("--limit", type=int, default=None, help="Use the first N training clips. Omit to use all 10000.")
    parser.add_argument(
        "--val-limit",
        type=int,
        default=None,
        help="Score the first N validation clips. Defaults to --limit when that is set, otherwise all 2500.",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-clips", type=int, default=DEFAULT_BATCH_CLIPS)
    parser.add_argument("--jobs", type=int, default=-1, help="Worker processes for feature extraction. -1 uses every CPU.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    jobs = resolve_jobs(args.jobs)
    val_limit = args.limit if args.val_limit is None else args.val_limit
    started = time.perf_counter()
    train_audio, train_names, train_labels = select_clips("train", args.limit)
    val_audio, val_names, val_labels = select_clips("validation", val_limit)
    if train_audio.resolve() == val_audio.resolve():
        raise RuntimeError("train and validation audio directories are the same")
    smoke = len(train_names) != EXPECTED_CLIPS["train"] or len(val_names) != EXPECTED_CLIPS["validation"]
    train_features, train_targets = build_feature_cache("train", train_audio, train_names, train_labels, jobs)
    val_features, _val_targets = build_feature_cache("validation", val_audio, val_names, val_labels, jobs)
    models, mean, std, positives = fit_classifiers(
        train_features,
        train_targets,
        epochs=args.epochs,
        batch_clips=args.batch_clips,
        seed=args.seed,
    )
    probabilities = predict_probabilities(models, val_features, mean, std, args.batch_clips)
    segment_totals, event_totals, first_reference, first_prediction = score_split(val_names, val_labels, probabilities)
    command_bits = ["uv run python -m nndl_project.baseline"]
    if args.limit is not None:
        command_bits.append(f"--limit {args.limit}")
    if args.val_limit is not None:
        command_bits.append(f"--val-limit {args.val_limit}")
    if args.epochs != DEFAULT_EPOCHS:
        command_bits.append(f"--epochs {args.epochs}")
    if args.batch_clips != DEFAULT_BATCH_CLIPS:
        command_bits.append(f"--batch-clips {args.batch_clips}")
    if args.jobs != -1:
        command_bits.append(f"--jobs {args.jobs}")
    if args.seed != 0:
        command_bits.append(f"--seed {args.seed}")
    report = project_root() / "reports" / ("baseline_logistic_smoke.md" if smoke else "baseline_logistic.md")
    write_report(
        report,
        command=" ".join(command_bits),
        smoke=smoke,
        train_names=train_names,
        val_names=val_names,
        positives=positives,
        segment_totals=segment_totals,
        event_totals=event_totals,
        first_name=val_names[0],
        first_reference=first_reference,
        first_prediction=first_prediction,
        elapsed_seconds=time.perf_counter() - started,
        epochs=args.epochs,
        batch_clips=args.batch_clips,
        seed=args.seed,
    )
    model_path = _cache_dir("train", len(train_names)) / ("models_smoke.joblib" if smoke else "models.joblib")
    joblib.dump(
        {
            "classes": CLASSES,
            "models": models,
            "mean": mean,
            "std": std,
            "threshold": THRESHOLD,
            "median_frames": MEDIAN_FRAMES,
            "alpha": SGD_ALPHA,
            "epochs": args.epochs,
            "seed": args.seed,
        },
        model_path,
    )
    print(f"DONE: wrote {report}", flush=True)
    return report


def main() -> None:
    run()


if __name__ == "__main__":
    main()
