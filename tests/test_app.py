"""The web frontend serves its page and returns events for an uploaded wav."""

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

torch = pytest.importorskip("torch")

from nndl_project.app import Detector, make_handler  # noqa: E402
from nndl_project.constants import CLASSES, N_MELS  # noqa: E402
from nndl_project.crnn import SmallCRNN  # noqa: E402


@pytest.fixture()
def server(tmp_path: Path):
    torch.manual_seed(0)
    model = SmallCRNN()
    with torch.no_grad():
        model.head.bias.fill_(-8.0)
        model.head.bias[CLASSES.index("Dog")] = 8.0
    checkpoint = tmp_path / "m.pt"
    torch.save(
        {
            "epoch": 1,
            "model_state": model.state_dict(),
            "mean": np.zeros(N_MELS, dtype=np.float32),
            "std": np.ones(N_MELS, dtype=np.float32),
            "identity": {"model": "small_crnn", "channels": 16, "hidden": 64},
        },
        checkpoint,
    )
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(Detector(checkpoint, None)))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def test_page_and_detection(server: str, tmp_path: Path) -> None:
    page = urllib.request.urlopen(server + "/").read().decode()
    assert "Urban sound event detection" in page
    wav = tmp_path / "clip.wav"
    sf.write(wav, 0.05 * np.random.default_rng(0).standard_normal(160_000).astype(np.float32), 16_000)
    request = urllib.request.Request(
        server + "/detect",
        data=wav.read_bytes(),
        headers={"X-Filename": "clip.wav", "X-Street": "1"},
        method="POST",
    )
    result = json.loads(urllib.request.urlopen(request).read())
    assert [e["label"] for e in result["events"]] == ["dog"]
    assert result["reference"] is None


def test_bad_upload_is_an_error_not_a_crash(server: str) -> None:
    request = urllib.request.Request(server + "/detect", data=b"not audio", method="POST")
    with pytest.raises(urllib.error.HTTPError) as caught:
        urllib.request.urlopen(request)
    assert caught.value.code == 400
    # The server still answers afterwards.
    assert urllib.request.urlopen(server + "/info").status == 200
