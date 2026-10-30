"""Tune the decode on validation, then score the trained model on the held-out sets.

This is the last step of the method in the problem statement.

1. Frame probabilities on all 2500 synthetic21_validation clips.
2. Score them with the fixed decode (median 11 frames, threshold 0.5). This
   must reproduce the number in the model's own report, and it is printed
   next to that number as a check.
3. Tune the decode on validation only: one threshold per class and one
   median-filter length, chosen to maximize event F1. Validation is the
   development split, so this is where tuning is allowed.
4. Freeze those settings. Score the held-out sets with both the fixed and the
   tuned decode:
   - DESED public evaluation, 692 real clips (the real test set).
   - The 16 DCASE 2019 synthetic evaluation folders (robustness test).
   Nothing from these sets is used to choose anything.
5. Report every score twice: with the ten DESED names, and after the fixed
   street map in label_map.py (speech_crowd, dog, other). The seven street
   classes with no timestamp labels are listed and not scored.

Run from the project directory:

    uv run python -m nndl_project.evaluate                  # everything
    uv run python -m nndl_project.evaluate --quick          # public test + fbsnr_30dB only
    uv run python -m nndl_project.evaluate --model frame_gru

Output: reports/final_evaluation_<model>.md and .json. The json holds the
tuned thresholds, which `nndl_project.predict --thresholds` reads.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.ndimage import median_filter

from nndl_project.baseline import build_feature_cache, resolve_jobs, select_clips
from nndl_project.constants import CLASSES, CLIP_SECONDS, MEDIAN_FRAMES, THRESHOLD
from nndl_project.data import Event, list_clip_names, project_root
from nndl_project.decode import decode_events
from nndl_project.features import load_any_16k, log_mel
from nndl_project.label_map import (
    DESED_TO_STREET,
    STREET_CLASSES,
    STREET_CLASSES_WITH_TIMESTAMPS,
    STREET_CLASSES_WITHOUT_TIMESTAMPS,
    check_map,
    to_street_events,
)
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
from nndl_project.predict import LoadedModel, default_checkpoint, load_model

THRESHOLD_GRID: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
MEDIAN_GRID: tuple[int, ...] = (5, 11, 21, 31)

DCASE2019_FOLDERS: tuple[str, ...] = (
    "fbsnr_30dB",
    "fbsnr_24dB",
    "fbsnr_15dB",
    "fbsnr_0dB",
    "distorted_clipping",
    "distorted_drc",
    "distorted_highpass_filter",
    "distorted_lowpass_filter",
    "distorted_smartphone_playback",
    "distorted_smartphone_recording",
    "500ms",
    "5500ms",
    "9500ms",
    "ls_0dB",
    "ls_15dB",
    "ls_30dB",
)


# ----------------------------------------------------------------------------
# Held-out data
# ----------------------------------------------------------------------------


def heldout_sets(quick: bool) -> list[tuple[str, Path, Path]]:
    """Name, wav folder, and label table for each held-out set."""
    desed = project_root() / "data" / "raw" / "desed"
    sets = [("public_eval", desed / "audio" / "eval" / "public", desed / "metadata" / "eval" / "public.tsv")]
    synth = desed / "dcase2019" / "dataset"
    folders = ("fbsnr_30dB",) if quick else DCASE2019_FOLDERS
    for folder in folders:
        sets.append(
            (
                f"dcase2019/{folder}",
                synth / "audio" / "eval" / folder,
                synth / "metadata" / "eval" / f"{folder}.tsv",
            )
        )
    return sets


def read_labels_lenient(tsv: Path) -> tuple[dict[str, list[Event]], int]:
    """Read a held-out strong-label table.

    The synthetic training tables are clean and the strict reader refuses
    anything odd. Held-out tables are read more gently: `1.jams` is the label
    for `1.wav`, a row with no label marks a clip with no events, offsets are
    cut at 10 s, and a row that still makes no sense is counted and dropped.
    """
    labels: dict[str, list[Event]] = {}
    dropped = 0
    with tsv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            name = (row.get("filename") or "").strip()
            if not name:
                dropped += 1
                continue
            if name.endswith(".jams"):
                name = name[: -len(".jams")] + ".wav"
            events = labels.setdefault(name, [])
            label = (row.get("event_label") or "").strip()
            if label in {"", "NaN", "nan"}:
                continue
            try:
                onset = float(row["onset"])
                offset = min(float(row["offset"]), CLIP_SECONDS)
            except (TypeError, ValueError, KeyError):
                dropped += 1
                continue
            if label not in CLASSES or not np.isfinite(onset) or offset <= onset or onset < 0:
                dropped += 1
                continue
            events.append(Event(label, onset, offset))
    return labels, dropped


def _features_for(path: Path) -> tuple[np.ndarray, float]:
    waveform, seconds = load_any_16k(path)
    return log_mel(waveform), seconds


# ----------------------------------------------------------------------------
# Probabilities, decode, and scores
# ----------------------------------------------------------------------------


def batched_probabilities(model: LoadedModel, features: np.ndarray, batch: int = 32) -> np.ndarray:
    out = np.empty((features.shape[0], features.shape[1], len(CLASSES)), dtype=np.float32)
    for start in range(0, features.shape[0], batch):
        block = np.asarray(features[start : start + batch], dtype=np.float32)
        out[start : start + batch] = model.probabilities(block)
    return out


def smooth_all(probabilities: np.ndarray, median_frames: int) -> np.ndarray:
    """Median-filter every clip along time. The window never crosses clips."""
    if median_frames == 1:
        return probabilities
    return median_filter(probabilities, size=(1, median_frames, 1), mode="nearest").astype(np.float32)


def decode_smoothed(smoothed: np.ndarray, thresholds: np.ndarray) -> list[Event]:
    # The probabilities are already smoothed, so the decode filter is 1 frame.
    return decode_events(smoothed, threshold=thresholds, median_frames=1)


def _trim(events: list[Event], seconds: float) -> list[Event]:
    """Drop or cut predicted events that fall in the padding of a short clip."""
    limit = min(seconds, CLIP_SECONDS)
    trimmed = []
    for e in events:
        if e.onset >= limit:
            continue
        trimmed.append(Event(e.label, e.onset, min(e.offset, limit), e.score))
    return trimmed


def score_clips(
    references: list[list[Event]],
    predictions: list[list[Event]],
) -> dict[str, dict[str, dict[str, Counts]]]:
    """Segment and event counts, with the DESED names and with the street map."""
    out = {
        "desed": {"segment": empty_counts(CLASSES), "event": empty_counts(CLASSES)},
        "street": {"segment": empty_counts(STREET_CLASSES), "event": empty_counts(STREET_CLASSES)},
    }
    for reference, prediction in zip(references, predictions, strict=True):
        accumulate(out["desed"]["segment"], segment_counts_for_clip(reference, prediction))
        accumulate(out["desed"]["event"], event_counts_for_clip(reference, prediction))
        ref_street = to_street_events(reference)
        pred_street = to_street_events(prediction)
        accumulate(
            out["street"]["segment"],
            segment_counts_for_clip(ref_street, pred_street, classes=STREET_CLASSES),
        )
        accumulate(
            out["street"]["event"],
            event_counts_for_clip(ref_street, pred_street, classes=STREET_CLASSES),
        )
    return out


def summarize(counts: dict[str, Counts], classes: tuple[str, ...]) -> dict[str, object]:
    per_class = {}
    for name in classes:
        c = counts[name]
        prf = precision_recall_f1(c)
        per_class[name] = {
            "tp": c.tp,
            "fp": c.fp,
            "fn": c.fn,
            "precision": None if prf is None else prf[0],
            "recall": None if prf is None else prf[1],
            "f1": None if prf is None else prf[2],
        }
    micro = precision_recall_f1(micro_counts({n: counts[n] for n in classes}))
    return {
        "per_class": per_class,
        "micro_f1": None if micro is None else micro[2],
        "macro_f1": macro_f1(counts, classes),
    }


def full_summary(scored: dict[str, dict[str, dict[str, Counts]]]) -> dict[str, object]:
    street_scored = STREET_CLASSES_WITH_TIMESTAMPS
    return {
        "desed": {kind: summarize(scored["desed"][kind], CLASSES) for kind in ("segment", "event")},
        # Only street classes that receive DESED labels are scored. The other
        # seven have no timestamp labels, so their F1 would be undefined.
        "street": {kind: summarize(scored["street"][kind], street_scored) for kind in ("segment", "event")},
    }


def tune_decode(probabilities: np.ndarray, references: list[list[Event]]) -> dict[str, object]:
    """Per-class threshold and one median length that maximize validation event F1.

    Classes are decoded independently, so one pass per (median, threshold)
    gives the counts for every class at once. For each median length, each
    class takes its best threshold (ties go to the value nearest 0.5). The
    median length with the best event macro F1 over those choices wins.
    Ties between median lengths go to the shorter one.
    """
    best: dict[str, object] | None = None
    trace = []
    for median in MEDIAN_GRID:
        smoothed = smooth_all(probabilities, median)
        per_threshold: dict[float, dict[str, Counts]] = {}
        for threshold in THRESHOLD_GRID:
            total = empty_counts(CLASSES)
            thresholds = np.full(len(CLASSES), threshold)
            for index, reference in enumerate(references):
                prediction = decode_smoothed(smoothed[index], thresholds)
                accumulate(total, event_counts_for_clip(reference, prediction))
            per_threshold[threshold] = total
        chosen: dict[str, float] = {}
        chosen_f1: dict[str, float] = {}
        for name in CLASSES:
            scores = []
            for threshold in THRESHOLD_GRID:
                prf = precision_recall_f1(per_threshold[threshold][name])
                scores.append(-1.0 if prf is None else prf[2])
            # Highest F1. On a tie, the threshold nearest 0.5 wins, so a class
            # that scores 0 everywhere keeps the untuned default.
            pick = max(range(len(THRESHOLD_GRID)), key=lambda i: (scores[i], -abs(THRESHOLD_GRID[i] - THRESHOLD)))
            chosen[name] = THRESHOLD_GRID[pick]
            chosen_f1[name] = scores[pick]
        defined = [v for v in chosen_f1.values() if v >= 0.0]
        macro = float(np.mean(defined)) if defined else 0.0
        trace.append({"median_frames": median, "event_macro_f1": macro, "thresholds": chosen})
        print(f"  median {median:2d} frames: best per-class thresholds give validation event macro F1 {macro:.3f}")
        if best is None or macro > float(best["event_macro_f1"]):  # type: ignore[arg-type]
            best = {"median_frames": median, "thresholds": chosen, "event_macro_f1": macro}
    assert best is not None
    best["search"] = trace
    best["threshold_grid"] = list(THRESHOLD_GRID)
    best["median_grid"] = list(MEDIAN_GRID)
    return best


def predictions_for(
    probabilities: np.ndarray,
    median_frames: int,
    thresholds: np.ndarray,
    seconds: list[float] | None = None,
) -> list[list[Event]]:
    smoothed = smooth_all(probabilities, median_frames)
    out = []
    for index in range(smoothed.shape[0]):
        events = decode_smoothed(smoothed[index], thresholds)
        if seconds is not None:
            events = _trim(events, seconds[index])
        out.append(events)
    return out


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------


def _f(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _headline_rows(name: str, decode: str, summary: dict[str, object]) -> str:
    d = summary["desed"]
    s = summary["street"]
    return (
        f"| {name} | {decode} | {_f(d['segment']['micro_f1'])} | {_f(d['segment']['macro_f1'])} | "
        f"{_f(d['event']['micro_f1'])} | {_f(d['event']['macro_f1'])} | "
        f"{_f(s['segment']['macro_f1'])} | {_f(s['event']['macro_f1'])} |"
    )


def _class_table(summary: dict[str, object], space: str, classes: tuple[str, ...]) -> str:
    seg = summary[space]["segment"]["per_class"]
    evt = summary[space]["event"]["per_class"]
    lines = [
        "| class | segment F1 | event precision | event recall | event F1 | reference events |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in classes:
        e = evt[name]
        lines.append(
            f"| {name} | {_f(seg[name]['f1'])} | {_f(e['precision'])} | {_f(e['recall'])} | "
            f"{_f(e['f1'])} | {e['tp'] + e['fn']} |"
        )
    return "\n".join(lines)


def render_report(path: Path, record: dict[str, object]) -> None:
    tuned = record["tuned_decode"]
    results = record["results"]
    rows = []
    for name, entry in results.items():
        rows.append(_headline_rows(name, "fixed 0.5 / 11", entry["fixed"]))
        rows.append(_headline_rows(name, "tuned", entry["tuned"]))
    threshold_line = ", ".join(f"{k} {v}" for k, v in tuned["thresholds"].items())
    public = results.get("public_eval")
    street_missing = ", ".join(STREET_CLASSES_WITHOUT_TIMESTAMPS)
    text = f"""# Final evaluation: {record["model"]}

Command: `{record["command"]}`

Checkpoint: `{record["checkpoint"]}` (epoch {record["epoch"]}). Finished {record["finished_at"]}, {record["elapsed_seconds"]:.0f} s.

## What was tuned, and on what

The decode was tuned on synthetic21_validation only. The held-out sets below did not choose anything.

- Median filter: {tuned["median_frames"]} frames, chosen from {tuned["median_grid"]}.
- Per-class thresholds, chosen from {tuned["threshold_grid"]}: {threshold_line}.
- Validation event macro F1 with these settings: {tuned["event_macro_f1"]:.3f}. That number is fitted to validation and is optimistic. The held-out rows are the honest ones.

Check: validation with the fixed decode gives event macro F1 {_f(results["validation"]["fixed"]["desed"]["event"]["macro_f1"])}. The training report for this checkpoint should show the same number at its last epoch.

## Headline scores

"DESED" columns use the ten DESED classes. "Street" columns use the fixed map in `src/nndl_project/label_map.py`: Speech to speech_crowd, Dog to dog, the other eight to other. Only those three street classes have timestamp labels. {street_missing} have none in this project, so they are not scored and have no target (success criterion 4).

| set | decode | DESED segment micro F1 | DESED segment macro F1 | DESED event micro F1 | DESED event macro F1 | street segment macro F1 | street event macro F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

Clip counts: {", ".join(f"{k} {v['clips']}" for k, v in results.items())}.
"""
    if public is not None:
        text += f"""
## Public evaluation, per class, tuned decode

These are the 692 real clips. The model never trained on real audio, so the drop from validation is the synthetic-to-real gap.

{_class_table(public["tuned"], "desed", CLASSES)}

Street classes:

{_class_table(public["tuned"], "street", STREET_CLASSES_WITH_TIMESTAMPS)}

Street classes with no timestamp labels, not scored: {street_missing}.
"""
    notes = record.get("notes") or []
    if notes:
        text += "\n## Notes\n\n" + "\n".join(f"- {n}" for n in notes) + "\n"
    text += f"""
## Label map

| DESED class | street class |
| --- | --- |
{chr(10).join(f"| {k} | {v} |" for k, v in DESED_TO_STREET.items())}
"""
    path.write_text(text, encoding="utf-8")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def run(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=["crnn", "frame_gru"], default="crnn")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--quick", action="store_true", help="public eval and fbsnr_30dB only")
    parser.add_argument("--val-limit", type=int, default=None, help="smoke check on the first N validation clips")
    parser.add_argument("--jobs", type=int, default=-1, help="parallel feature workers, -1 for all cores")
    args = parser.parse_args(argv)

    check_map()
    started = time.time()
    jobs = resolve_jobs(args.jobs)
    checkpoint = args.checkpoint or default_checkpoint(args.model)
    model = load_model(checkpoint)
    print(f"Loaded {model.name} epoch {model.epoch} from {checkpoint}")
    notes: list[str] = []

    # 1. Validation probabilities. The log-mel cache is reused when present.
    audio_dir, names, labels = select_clips("validation", args.val_limit)
    features, _targets = build_feature_cache("validation", audio_dir, names, labels, jobs)
    val_probs = batched_probabilities(model, features)
    val_refs = [labels.get(name, []) for name in names]
    print(f"Validation: {len(names)} clips scored by the network")

    # 2. Fixed decode, the setting every training report used.
    fixed = np.full(len(CLASSES), THRESHOLD)
    results: dict[str, dict[str, object]] = {}
    val_fixed = full_summary(score_clips(val_refs, predictions_for(val_probs, MEDIAN_FRAMES, fixed)))

    # 3. Tune on validation.
    print("Tuning the decode on validation")
    tuned = tune_decode(val_probs, val_refs)
    tuned_thresholds = np.array([tuned["thresholds"][n] for n in CLASSES])
    tuned_median = int(tuned["median_frames"])
    val_tuned = full_summary(score_clips(val_refs, predictions_for(val_probs, tuned_median, tuned_thresholds)))
    results["validation"] = {"clips": len(names), "fixed": val_fixed, "tuned": val_tuned}
    del val_probs

    # 4. Held-out sets, scored once with frozen settings.
    for set_name, wav_dir, tsv in heldout_sets(args.quick):
        if not wav_dir.is_dir() or not tsv.is_file():
            note = f"{set_name}: skipped, {wav_dir if not wav_dir.is_dir() else tsv} is missing."
            print(note)
            notes.append(note)
            continue
        set_labels, dropped = read_labels_lenient(tsv)
        wavs = list_clip_names(wav_dir)
        unlabeled = [w for w in wavs if w not in set_labels]
        missing_audio = [n for n in set_labels if n not in set(wavs)]
        if dropped or unlabeled or missing_audio:
            notes.append(
                f"{set_name}: {dropped} label rows dropped as unusable, {len(unlabeled)} wavs without a "
                f"label row (scored as having no events), {len(missing_audio)} labeled clips without a wav (skipped)."
            )
        print(f"{set_name}: extracting log-mel for {len(wavs)} clips")
        loaded = Parallel(n_jobs=jobs)(delayed(_features_for)(wav_dir / w) for w in wavs)
        feats = np.stack([f for f, _s in loaded])
        seconds = [s for _f, s in loaded]
        short = sum(1 for s in seconds if s < CLIP_SECONDS - 0.05)
        if short:
            notes.append(f"{set_name}: {short} clips shorter than 10 s were padded; events in the padding were dropped.")
        probs = batched_probabilities(model, feats)
        refs = [set_labels.get(w, []) for w in wavs]
        entry = {
            "clips": len(wavs),
            "fixed": full_summary(score_clips(refs, predictions_for(probs, MEDIAN_FRAMES, fixed, seconds))),
            "tuned": full_summary(score_clips(refs, predictions_for(probs, tuned_median, tuned_thresholds, seconds))),
        }
        results[set_name] = entry
        print(
            f"  event macro F1 fixed {_f(entry['fixed']['desed']['event']['macro_f1'])}, "
            f"tuned {_f(entry['tuned']['desed']['event']['macro_f1'])}"
        )

    reports = project_root() / "reports"
    suffix = model.name.replace("small_", "")
    if args.val_limit is not None or args.quick:
        suffix += "_smoke" if args.val_limit is not None else "_quick"
    md_path = reports / f"final_evaluation_{suffix}.md"
    json_path = reports / f"final_evaluation_{suffix}.json"
    record = {
        "model": model.name,
        "epoch": model.epoch,
        "checkpoint": str(checkpoint),
        "command": "uv run python -m nndl_project.evaluate " + " ".join(argv if argv is not None else sys.argv[1:]),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_seconds": time.time() - started,
        "tuned_decode": tuned,
        "label_map": DESED_TO_STREET,
        "street_classes_without_timestamps": list(STREET_CLASSES_WITHOUT_TIMESTAMPS),
        "results": results,
        "notes": notes,
    }
    json_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    render_report(md_path, record)
    print(f"Wrote {md_path} and {json_path}")
    return md_path


def main() -> None:
    run()


if __name__ == "__main__":
    main()
