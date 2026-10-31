"""A small web page for the sound event detector.

Run from the project directory:

    uv run python -m nndl_project.app

then open http://localhost:8000 in a browser. Choose or drop a wav file. The
page plays the clip and shows the detected events as a table and as a
timeline, one lane per class. If the file is one of the 692 DESED public
test clips, the page also shows the reference events for comparison.

There are no extra dependencies: the server is Python's built-in http.server
and the page is plain HTML and JavaScript. The model is the CRNN checkpoint.
If `nndl_project.evaluate` has been run, its tuned thresholds are used.
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
from nndl_project.data import Event, project_root
from nndl_project.label_map import STREET_CLASSES, to_street_events
from nndl_project.predict import (
    LoadedModel,
    default_checkpoint,
    events_as_dicts,
    load_model,
    predict_clip,
    read_thresholds,
)

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
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self) -> None:  # noqa: N802
            if not self.path.startswith("/detect"):
                self._send(404, b"not found", "text/plain")
                return
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
    detector = Detector(checkpoint, args.thresholds or _default_thresholds())
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(detector))
    url = f"http://localhost:{args.port}"
    print(f"Model: {detector.model.name}, epoch {detector.model.epoch} ({checkpoint})")
    print(f"Decode: {detector.decode_note}")
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


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Urban sound event detection</title>
<style>
  :root { --ink:#1a202c; --muted:#5a6475; --line:#e2e6ec; --bg:#f6f7f9; --accent:#2b6cb0; }
  * { box-sizing:border-box; }
  body { margin:0; font:15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; color:var(--ink); background:var(--bg); }
  header { background:#fff; border-bottom:1px solid var(--line); padding:18px 32px; }
  header h1 { margin:0; font-size:20px; }
  header p { margin:2px 0 0; color:var(--muted); font-size:13px; }
  main { max-width:1100px; margin:24px auto; padding:0 24px; }
  .card { background:#fff; border:1px solid var(--line); border-radius:10px; padding:20px 24px; margin-bottom:20px; }
  #drop { border:2px dashed #b8c2d0; border-radius:10px; padding:28px; text-align:center; color:var(--muted); cursor:pointer; }
  #drop.over { border-color:var(--accent); background:#ebf4ff; }
  .row { display:flex; gap:16px; align-items:center; flex-wrap:wrap; margin-top:14px; }
  button { background:var(--accent); color:#fff; border:0; border-radius:6px; padding:9px 18px; font-size:14px; cursor:pointer; }
  button:disabled { background:#a0aec0; cursor:default; }
  label.toggle { display:flex; gap:6px; align-items:center; color:var(--muted); font-size:14px; }
  audio { width:100%; margin-top:6px; }
  h2 { font-size:16px; margin:0 0 12px; }
  .meta { color:var(--muted); font-size:13px; }
  table { border-collapse:collapse; width:100%; font-size:14px; }
  th, td { text-align:left; padding:7px 10px; border-bottom:1px solid var(--line); }
  th { color:var(--muted); font-weight:600; font-size:12px; text-transform:uppercase; letter-spacing:.03em; }
  td.num { font-variant-numeric:tabular-nums; }
  .chip { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:7px; vertical-align:middle; }
  .bar { height:8px; background:#e2e8f0; border-radius:4px; width:90px; display:inline-block; vertical-align:middle; margin-left:8px; }
  .bar span { display:block; height:100%; border-radius:4px; background:var(--accent); }
  .error { color:#c53030; }
  svg text { font-family:inherit; }
  .legend { display:flex; gap:18px; font-size:12px; color:var(--muted); margin-top:8px; }
  .legend i { display:inline-block; width:22px; height:10px; margin-right:6px; vertical-align:middle; border-radius:2px; }
</style>
</head>
<body>
<header>
  <h1>Urban sound event detection with timestamps</h1>
  <p id="info">Loading model…</p>
</header>
<main>
  <div class="card">
    <div id="drop">Drop a .wav file here, or click to choose one<br><span class="meta">About 10 seconds, any sample rate, mono or stereo</span></div>
    <input id="file" type="file" accept=".wav,audio/wav" hidden>
    <div class="row">
      <button id="run" disabled>Detect events</button>
      <label class="toggle"><input id="street" type="checkbox" checked> Street class names</label>
      <span id="status" class="meta"></span>
    </div>
    <audio id="player" controls hidden></audio>
  </div>

  <div class="card" id="timelineCard" hidden>
    <h2>Timeline</h2>
    <div id="timeline"></div>
    <div class="legend" id="legend"></div>
  </div>

  <div class="card" id="tableCard" hidden>
    <h2>Detected events</h2>
    <p class="meta" id="summary"></p>
    <table>
      <thead><tr><th>Class</th><th>Start (s)</th><th>End (s)</th><th>Duration (s)</th><th>Score</th></tr></thead>
      <tbody id="rows"></tbody>
    </table>
  </div>
</main>
<script>
const COLORS = ["#2b6cb0","#c05621","#2f855a","#6b46c1","#b83280","#2c7a7b","#975a16","#c53030","#4a5568","#d69e2e"];
let file = null, result = null;
const $ = id => document.getElementById(id);

fetch("/info").then(r => r.json()).then(i => { $("info").textContent = "Model: " + i.model + " · " + i.decode; });

function choose(f) {
  if (!f) return;
  file = f;
  $("run").disabled = false;
  $("status").textContent = f.name;
  const p = $("player"); p.src = URL.createObjectURL(f); p.hidden = false;
}
$("drop").onclick = () => $("file").click();
$("file").onchange = e => choose(e.target.files[0]);
$("drop").ondragover = e => { e.preventDefault(); $("drop").classList.add("over"); };
$("drop").ondragleave = () => $("drop").classList.remove("over");
$("drop").ondrop = e => { e.preventDefault(); $("drop").classList.remove("over"); choose(e.dataTransfer.files[0]); };
$("street").onchange = () => { if (file) detect(); };
$("run").onclick = detect;

async function detect() {
  $("run").disabled = true;
  $("status").textContent = "Running the model on " + file.name + "…";
  try {
    const r = await fetch("/detect", { method: "POST", body: file,
      headers: { "X-Filename": encodeURIComponent(file.name), "X-Street": $("street").checked ? "1" : "0" } });
    const data = await r.json();
    if (!r.ok) throw new Error(data.error || r.statusText);
    result = data; render();
    $("status").textContent = file.name;
  } catch (err) {
    $("status").innerHTML = '<span class="error">' + err.message + '</span>';
  }
  $("run").disabled = false;
}

let lanes = [];
function colorOf(label) { const i = lanes.indexOf(label); return COLORS[(i < 0 ? 0 : i) % COLORS.length]; }

function render() {
  const ev = result.events, ref = result.reference;
  lanes = result.classes.filter(c => ev.some(e => e.label === c) || (ref && ref.some(e => e.label === c)));
  $("summary").textContent = ev.length + " event" + (ev.length === 1 ? "" : "s") + " detected" +
    (ref ? " · reference has " + ref.length + " (this is a DESED public test clip)" : "");
  $("rows").innerHTML = ev.length ? ev.map(e =>
    `<tr><td><span class="chip" style="background:${colorOf(e.label)}"></span>${e.label}</td>
     <td class="num">${e.start.toFixed(3)}</td><td class="num">${e.end.toFixed(3)}</td>
     <td class="num">${(e.end - e.start).toFixed(3)}</td>
     <td class="num">${e.score.toFixed(3)}<span class="bar"><span style="width:${Math.round(e.score*100)}%;background:${colorOf(e.label)}"></span></span></td></tr>`
  ).join("") : '<tr><td colspan="5" class="meta">No event above the threshold.</td></tr>';
  $("tableCard").hidden = false;
  drawTimeline(ev, ref);
}

function drawTimeline(ev, ref) {
  const shown = lanes.length ? lanes : result.classes.slice(0, 3);
  const W = 1000, left = 150, right = 20, laneH = ref ? 46 : 30, top = 10, axisH = 28;
  const H = top + shown.length * laneH + axisH;
  const x = t => left + (W - left - right) * t / 10;
  let s = `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="Event timeline">`;
  for (let t = 0; t <= 10; t++) {
    s += `<line x1="${x(t)}" x2="${x(t)}" y1="${top}" y2="${H - axisH}" stroke="#edf0f4"/>`;
    s += `<text x="${x(t)}" y="${H - 8}" font-size="12" fill="#5a6475" text-anchor="middle">${t}s</text>`;
  }
  shown.forEach((c, i) => {
    const y = top + i * laneH;
    s += `<text x="${left - 10}" y="${y + laneH / 2 + 4}" font-size="13" fill="#1a202c" text-anchor="end">${c}</text>`;
    s += `<line x1="${left}" x2="${W - right}" y1="${y + laneH}" y2="${y + laneH}" stroke="#e2e6ec"/>`;
    const predY = ref ? y + 6 : y + 6, h = ref ? 16 : 18;
    if (ref) ref.filter(e => e.label === c).forEach(e => {
      s += `<rect x="${x(e.start)}" y="${y + 26}" width="${Math.max(2, x(e.end) - x(e.start))}" height="12" rx="2" fill="none" stroke="${colorOf(c)}" stroke-width="2" stroke-dasharray="4 2"><title>reference ${e.start}–${e.end} s</title></rect>`;
    });
    ev.filter(e => e.label === c).forEach(e => {
      s += `<rect x="${x(e.start)}" y="${predY}" width="${Math.max(2, x(e.end) - x(e.start))}" height="${h}" rx="3" fill="${colorOf(c)}" fill-opacity="${0.35 + 0.65 * e.score}"><title>${c} ${e.start}–${e.end} s, score ${e.score}</title></rect>`;
    });
  });
  s += `<line id="cursor" x1="${left}" x2="${left}" y1="${top}" y2="${H - axisH}" stroke="#e53e3e" stroke-width="2" visibility="hidden"/>`;
  s += `</svg>`;
  $("timeline").innerHTML = s;
  $("legend").innerHTML = '<span><i style="background:#4a5568"></i>detected (darker = higher score)</span>' +
    (ref ? '<span><i style="border:2px dashed #4a5568"></i>reference (ground truth)</span>' : '') +
    '<span><i style="background:#e53e3e;height:3px"></i>playback position</span>';
  $("timelineCard").hidden = false;
  const p = $("player"), cur = $("cursor");
  p.ontimeupdate = () => { const t = Math.min(p.currentTime, 10); cur.setAttribute("x1", x(t)); cur.setAttribute("x2", x(t)); cur.setAttribute("visibility", "visible"); };
}
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
