"""Presentation page for the four trained sound-event models.

Run from the project directory:

    uv run python -m nndl_project.app

then open http://localhost:8000. The page shows the validation scores and
lets you drop in a wav. Each available model draws its events on one timeline.
A validation or public-test filename also draws the reference events.

There are no extra dependencies. The live comparison uses the fixed decode
from the score table: median 11 frames, threshold 0.5.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

from nndl_project.constants import CLASSES, MEDIAN_FRAMES, THRESHOLD
from nndl_project.data import Event, events_for_split, project_root
from nndl_project.features import load_any_16k, log_mel
from nndl_project.label_map import STREET_CLASSES, to_street_events
from nndl_project.predict import (
    LoadedModel,
    default_checkpoint,
    events_as_dicts,
    load_model,
    predict_clip,
    read_thresholds,
)
from nndl_project.present import events_for_features, example_wav, load_demo_models, scores_payload

MAX_UPLOAD_BYTES = 50 * 1024 * 1024


class Detector:
    """The loaded model, the decode settings, and the public-test labels."""

    def __init__(self, checkpoint: Path, thresholds: Path | None) -> None:
        self.model: LoadedModel = load_model(checkpoint)
        self.checkpoint = checkpoint
        self.threshold: float | list[float] = THRESHOLD
        self.median = MEDIAN_FRAMES
        self.decode_note = "fixed decode: threshold 0.5, median 11 frames"
        if thresholds is not None and thresholds.is_file():
            self.threshold, self.median = read_thresholds(thresholds)
            self.decode_note = f"tuned decode from {thresholds.name}, median {self.median} frames"
        self.lock = threading.Lock()
        self.public_labels = _public_labels()

    def detect(self, wav: Path, original_name: str, street: bool) -> dict[str, object]:
        with self.lock:
            events = predict_clip(
                self.model, wav, threshold=self.threshold, median_frames=self.median, street=street
            )
        reference: list[Event] | None = self.public_labels.get(original_name)
        if reference is not None and street:
            reference = to_street_events(reference)
        return {
            "file": original_name,
            "classes": list(STREET_CLASSES if street else CLASSES),
            "events": events_as_dicts(events),
            "reference": None if reference is None else events_as_dicts(reference),
            "model": f"{self.model.name}, epoch {self.model.epoch}",
            "decode": self.decode_note,
        }


def _public_labels() -> dict[str, list[Event]]:
    tsv = project_root() / "data" / "raw" / "desed" / "metadata" / "eval" / "public.tsv"
    if not tsv.is_file():
        return {}
    from nndl_project.evaluate import read_labels_lenient

    labels, _dropped = read_labels_lenient(tsv)
    return labels


def _validation_labels() -> dict[str, list[Event]]:
    try:
        return events_for_split("validation")
    except FileNotFoundError:
        return {}


class Gallery(Detector):
    """The presentation server: the scoreboard plus every model on one wav."""

    def __init__(self, checkpoint: Path, thresholds: Path | None) -> None:
        super().__init__(checkpoint, thresholds)
        self.demo_models = load_demo_models()
        self.validation_labels = _validation_labels()

    def reference_for(self, original_name: str) -> tuple[list[Event] | None, str]:
        name = Path(original_name).name
        public = self.public_labels.get(name)
        if public is not None:
            return public, "DESED public test clip"
        validation = self.validation_labels.get(name)
        if validation is not None:
            return validation, "synthetic21_validation clip"
        return None, ""

    def compare(self, wav: Path, original_name: str, street: bool) -> dict[str, object]:
        with self.lock:
            waveform, seconds = load_any_16k(wav)
            features = log_mel(waveform)
            reference, reference_note = self.reference_for(original_name)
            if reference is not None and street:
                reference = to_street_events(reference)
            models = []
            for model in self.demo_models:
                events = events_for_features(model, features, seconds, street=street)
                models.append(
                    {
                        "name": model.name,
                        "detail": model.detail,
                        "events": events_as_dicts(events),
                    }
                )
        classes = list(STREET_CLASSES if street else CLASSES)
        return {
            "file": original_name,
            "seconds": round(seconds, 3),
            "classes": classes,
            "models": models,
            "reference": None if reference is None else events_as_dicts(reference),
            "reference_note": reference_note,
            "decode": "fixed decode for every model: threshold 0.5, median 11 frames",
        }


def _default_thresholds() -> Path | None:
    reports = project_root() / "reports"
    for name in ("final_evaluation_crnn.json", "final_evaluation_crnn_quick.json"):
        if (reports / name).is_file():
            return reports / name
    return None


def make_handler(detector: Detector) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            pass

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path in ("/", "/index.html"):
                self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            elif self.path == "/info":
                info = {"model": f"{detector.model.name}, epoch {detector.model.epoch}", "decode": detector.decode_note}
                self._send(200, json.dumps(info).encode(), "application/json")
            elif self.path == "/scores":
                self._send(200, json.dumps(scores_payload()).encode(), "application/json")
            elif self.path == "/example":
                wav = example_wav()
                if not wav.is_file():
                    self._send(404, b"validation 2150.wav is not on this machine", "text/plain")
                    return
                self._send(200, wav.read_bytes(), "audio/wav")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self) -> None:  # noqa: N802
            if self.path.startswith("/compare") and hasattr(detector, "compare"):
                self._accept_wav("compare")
                return
            if not self.path.startswith("/detect"):
                self._send(404, b"not found", "text/plain")
                return
            self._accept_wav("detect")

        def _accept_wav(self, mode: str) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > MAX_UPLOAD_BYTES:
                self._send(400, b'{"error": "send one wav file under 50 MB"}', "application/json")
                return
            name = Path(unquote(self.headers.get("X-Filename") or "clip.wav")).name
            street = self.headers.get("X-Street", "0") == "1"
            body = self.rfile.read(length)
            with tempfile.TemporaryDirectory() as tmp:
                wav = Path(tmp) / "upload.wav"
                wav.write_bytes(body)
                try:
                    if mode == "compare":
                        result = detector.compare(wav, name, street)
                    else:
                        result = detector.detect(wav, name, street)
                except Exception as exc:  # a bad file must not stop the server
                    message = json.dumps({"error": f"could not read {name} as audio: {exc}"})
                    self._send(400, message.encode(), "application/json")
                    return
            self._send(200, json.dumps(result).encode(), "application/json")

    return Handler


def run(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--thresholds", type=Path, default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    checkpoint = args.checkpoint or default_checkpoint("crnn")
    if args.checkpoint is None and checkpoint.is_file():
        detector = Gallery(checkpoint, None)
    else:
        detector = Detector(checkpoint, args.thresholds or _default_thresholds())
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(detector))
    url = f"http://localhost:{args.port}"
    loaded = getattr(detector, "demo_models", None)
    if loaded:
        print("Models on the page: " + ", ".join(model.name for model in loaded))
    print(f"Decode: fixed threshold 0.5, median 11 frames")
    print(f"Public-test reference labels loaded for {len(detector.public_labels)} clips")
    print(f"Open {url}   (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> None:
    run()


PAGE = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Urban sound event detection</title>
<style>
  :root { --ink:#1a202c; --muted:#5a6475; --line:#e2e6ec; --bg:#f4f6f8; --accent:#234e70; --best:#e8f2ea; }
  * { box-sizing:border-box; }
  body { margin:0; font:16px/1.45 Georgia, "Iowan Old Style", "Palatino Linotype", serif; color:var(--ink); background:var(--bg); }
  header { background:#234e70; color:#fff; padding:22px 32px; }
  header h1 { margin:0; font-size:26px; font-weight:600; }
  header p { margin:6px 0 0; color:#d6e2ec; font-size:15px; font-family: "Segoe UI", sans-serif; }
  main { max-width:1100px; margin:22px auto; padding:0 20px 40px; }
  .card { background:#fff; border:1px solid var(--line); border-radius:8px; padding:18px 22px; margin-bottom:16px; }
  h2 { font-size:18px; margin:0 0 8px; }
  .meta { color:var(--muted); font-size:13px; font-family:"Segoe UI", sans-serif; }
  table { border-collapse:collapse; width:100%; font-size:14px; font-family:"Segoe UI", sans-serif; }
  th, td { text-align:left; padding:7px 8px; border-bottom:1px solid var(--line); }
  th { color:var(--muted); font-weight:600; font-size:12px; text-transform:uppercase; letter-spacing:.03em; }
  td.num, th.num { text-align:right; font-variant-numeric:tabular-nums; }
  tr.best { background:var(--best); font-weight:600; }
  .scroll { overflow-x:auto; }
  #drop { border:2px dashed #b8c2d0; border-radius:8px; padding:22px; text-align:center; color:var(--muted); cursor:pointer; font-family:"Segoe UI", sans-serif; }
  #drop.over { border-color:var(--accent); background:#eef5fb; }
  .row { display:flex; gap:12px; align-items:center; flex-wrap:wrap; margin-top:12px; }
  button { background:var(--accent); color:#fff; border:0; border-radius:6px; padding:9px 16px; font-size:14px; cursor:pointer; font-family:"Segoe UI", sans-serif; }
  button.secondary { background:#fff; color:var(--accent); border:1px solid var(--accent); }
  button:disabled { background:#a0aec0; border-color:#a0aec0; color:#fff; cursor:default; }
  label.toggle { display:flex; gap:6px; align-items:center; color:var(--muted); font-size:14px; font-family:"Segoe UI", sans-serif; }
  audio { width:100%; margin-top:10px; }
  .error { color:#9b2c2c; }
  svg text { font-family:"Segoe UI", sans-serif; }
  .legend { display:flex; gap:16px; flex-wrap:wrap; font-size:12px; color:var(--muted); margin-top:8px; font-family:"Segoe UI", sans-serif; }
  .legend i { display:inline-block; width:18px; height:10px; margin-right:6px; vertical-align:middle; border-radius:2px; }
  h3 { font-size:15px; margin:16px 0 6px; font-family:"Segoe UI", sans-serif; }
</style>
</head>
<body>
<header>
  <h1>Urban sound event detection with timestamps</h1>
  <p id="info">Four trained models. Validation scores are below. Drop in a recording to see what each model marks.</p>
</header>
<main>
  <div class="card">
    <h2>Validation scores</h2>
    <p class="meta" id="scoreNote">Loading the saved scores…</p>
    <div class="scroll"><table id="scoreTable"></table></div>
    <h3>Event F1 by class</h3>
    <p class="meta">Event F1 counts a match only when the start is within 0.2 seconds and the end is close as well. Segment F1, in the table above, only asks whether the sound is present in the right second.</p>
    <div class="scroll"><table id="classTable"></table></div>
  </div>

  <div class="card">
    <div id="drop">Drop a .wav file here, or click to choose one<br><span class="meta">About 10 seconds. Any sample rate. One or two channels. The button loads validation clip 2150.wav: a vacuum cleaner for the whole clip and three short stretches of speech. The CRNN matches every reference event on it.</span></div>
    <input id="file" type="file" accept=".wav,audio/wav,audio/*" hidden>
    <div class="row">
      <button id="run" disabled>Run all models</button>
      <button id="example" class="secondary" type="button">Load validation clip 2150.wav</button>
      <label class="toggle"><input id="street" type="checkbox"> Show street class names</label>
      <span id="status" class="meta"></span>
    </div>
    <audio id="player" controls hidden></audio>
  </div>

  <div class="card" id="timelineCard" hidden>
    <h2>What each model heard</h2>
    <p class="meta" id="clipNote"></p>
    <div id="timeline"></div>
    <div class="legend" id="legend"></div>
  </div>

  <div class="card" id="tableCard" hidden>
    <h2>Events</h2>
    <div id="tables"></div>
  </div>
</main>
<script>
const CLASS_COLORS = ["#234e70","#c05621","#2f855a","#6b46c1","#b83280","#2c7a7b","#975a16","#9b2c2c","#4a5568","#b7791f","#2b6cb0","#718096"];
const $ = id => document.getElementById(id);
let file = null, result = null, colorIndex = {};

fetch("/scores").then(r => r.json()).then(renderScores).catch(() => {
  $("scoreNote").textContent = "Could not load the saved scores.";
});

function renderScores(data) {
  $("scoreNote").textContent = data.set + ". " + data.decode + ". " + data.held_out;
  const cols = [
    ["segment_micro", "Segment micro F1"],
    ["segment_macro", "Segment macro F1"],
    ["event_micro", "Event micro F1"],
    ["event_macro", "Event macro F1"],
  ];
  let html = "<thead><tr><th>Model</th>" + cols.map(c => `<th class="num">${c[1]}</th>`).join("") + "</tr></thead><tbody>";
  data.rows.forEach(row => {
    const best = row.model === "CRNN" ? " class='best'" : "";
    html += `<tr${best}><td>${row.model}<br><span class="meta">${row.detail}</span></td>` +
      cols.map(c => `<td class="num">${Number(row[c[0]]).toFixed(3)}</td>`).join("") + "</tr>";
  });
  $("scoreTable").innerHTML = html + "</tbody>";
  const names = data.models;
  let cls = "<thead><tr><th>Class</th>" + names.map(n => `<th class="num">${n}</th>`).join("") + "</tr></thead><tbody>";
  Object.entries(data.class_event_f1).forEach(([name, values]) => {
    const top = Math.max(...values);
    cls += `<tr><td>${name}</td>` + values.map(v => `<td class="num"${v === top ? " style='font-weight:600'" : ""}>${Number(v).toFixed(3)}</td>`).join("") + "</tr>";
  });
  $("classTable").innerHTML = cls + "</tbody>";
}

function choose(f) {
  if (!f) return;
  file = f;
  $("run").disabled = false;
  $("status").textContent = f.name;
  const player = $("player");
  player.src = URL.createObjectURL(f);
  player.hidden = false;
}
$("drop").onclick = () => $("file").click();
$("file").onchange = e => choose(e.target.files[0]);
$("drop").ondragover = e => { e.preventDefault(); $("drop").classList.add("over"); };
$("drop").ondragleave = () => $("drop").classList.remove("over");
$("drop").ondrop = e => { e.preventDefault(); $("drop").classList.remove("over"); choose(e.dataTransfer.files[0]); };
$("street").onchange = () => { if (file && result) detect(); };
$("run").onclick = detect;
$("example").onclick = async () => {
  $("status").textContent = "Loading validation clip 2150.wav…";
  const response = await fetch("/example");
  if (!response.ok) { $("status").textContent = "That clip is not on this machine."; return; }
  const blob = await response.blob();
  choose(new File([blob], "2150.wav", {type: "audio/wav"}));
  detect();
};
$("example").click();

async function detect() {
  $("run").disabled = true;
  $("example").disabled = true;
  $("status").textContent = "Running the models on " + file.name + "…";
  try {
    const response = await fetch("/compare", { method: "POST", body: file, headers: {
      "X-Filename": encodeURIComponent(file.name),
      "X-Street": $("street").checked ? "1" : "0",
    }});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || response.statusText);
    result = data;
    render();
    $("status").textContent = file.name;
  } catch (err) {
    $("status").innerHTML = '<span class="error">' + err.message + "</span>";
  }
  $("run").disabled = false;
  $("example").disabled = false;
}

function colorOf(label) {
  if (!(label in colorIndex)) colorIndex[label] = Object.keys(colorIndex).length;
  return CLASS_COLORS[colorIndex[label] % CLASS_COLORS.length];
}

function render() {
  const models = result.models || [];
  const ref = result.reference;
  const labels = new Set();
  models.forEach(m => m.events.forEach(e => labels.add(e.label)));
  if (ref) ref.forEach(e => labels.add(e.label));
  const note = models.map(m => m.name + " " + m.events.length).join(" · ");
  $("clipNote").textContent = (result.reference_note ? result.reference_note + ". " : "") + result.decode + ". " + note;
  $("tables").innerHTML = models.map(m => {
    const rows = m.events.length ? m.events.map(e =>
      `<tr><td><span class="legend"><i style="background:${colorOf(e.label)}"></i></span>${e.label}</td>
       <td class="num">${e.start.toFixed(2)}</td><td class="num">${e.end.toFixed(2)}</td>
       <td class="num">${(e.end - e.start).toFixed(2)}</td><td class="num">${e.score.toFixed(3)}</td></tr>`
    ).join("") : '<tr><td colspan="5" class="meta">No event above the threshold.</td></tr>';
    return `<h3>${m.name} <span class="meta">${m.detail} · ${m.events.length} events</span></h3>
      <table><thead><tr><th>Class</th><th class="num">Start</th><th class="num">End</th><th class="num">Duration</th><th class="num">Score</th></tr></thead><tbody>${rows}</tbody></table>`;
  }).join("");
  $("tableCard").hidden = false;
  drawOverview(models, ref, [...labels]);
}

function drawOverview(models, ref, labels) {
  const lanes = (ref ? [{name: "Reference", events: ref, ref: true}] : []).concat(
    models.map(m => ({name: m.name, events: m.events, ref: false}))
  );
  const W = 1000, left = 168, right = 16, laneH = 42, top = 8, axisH = 26;
  const H = top + lanes.length * laneH + axisH;
  const x = t => left + (W - left - right) * Math.max(0, Math.min(t, 10)) / 10;
  let svg = `<svg id="chart" viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="Events from each model">`;
  for (let t = 0; t <= 10; t++) {
    svg += `<line x1="${x(t)}" x2="${x(t)}" y1="${top}" y2="${H - axisH}" stroke="#edf0f4"/>`;
    svg += `<text x="${x(t)}" y="${H - 6}" font-size="12" fill="#5a6475" text-anchor="middle">${t}s</text>`;
  }
  lanes.forEach((lane, i) => {
    const y = top + i * laneH;
    svg += `<text x="${left - 8}" y="${y + 25}" font-size="13" fill="#1a202c" text-anchor="end">${lane.name}</text>`;
    svg += `<line x1="${left}" x2="${W - right}" y1="${y + laneH}" y2="${y + laneH}" stroke="#e2e6ec"/>`;
    lane.events.forEach(e => {
      const width = Math.max(2, x(e.end) - x(e.start));
      const opacity = lane.ref ? 0.95 : (0.45 + 0.55 * (e.score == null ? 0.5 : e.score));
      const fill = colorOf(e.label);
      svg += `<rect x="${x(e.start)}" y="${y + 10}" width="${width}" height="20" rx="3" fill="${fill}" fill-opacity="${opacity}" stroke="${lane.ref ? "#1a202c" : "none"}" stroke-width="${lane.ref ? 1.5 : 0}"><title>${lane.name}: ${e.label} ${Number(e.start).toFixed(2)} to ${Number(e.end).toFixed(2)} s</title></rect>`;
      if (width > 54) {
        svg += `<text x="${x(e.start) + 4}" y="${y + 24}" font-size="11" fill="#fff">${e.label}</text>`;
      }
    });
  });
  svg += `<line id="cursor" x1="${left}" x2="${left}" y1="${top}" y2="${H - axisH}" stroke="#9b2c2c" stroke-width="2" visibility="hidden"/>`;
  svg += "</svg>";
  $("timeline").innerHTML = svg;
  const swatches = labels.map(label => `<span><i style="background:${colorOf(label)}"></i>${label}</span>`).join("");
  $("legend").innerHTML = swatches + '<span><i style="background:#9b2c2c;height:3px"></i>playback</span>';
  $("timelineCard").hidden = false;
  const player = $("player");
  const cursor = document.getElementById("cursor");
  player.ontimeupdate = () => {
    const t = Math.min(player.currentTime, 10);
    cursor.setAttribute("x1", x(t));
    cursor.setAttribute("x2", x(t));
    cursor.setAttribute("visibility", "visible");
  };
}
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
