"""Media locate: every dataset uri (condition or target) becomes one local file.

    resolve_media_uri (Dataset)
      "/abs/a.png" | "https://host/clip?id=1"     ──► unchanged
      "rel/a.png"                                  ──► anchored at the jsonl directory (not the symlink target)
      "s3://bucket/a.png"                          ──► unsupported scheme
    maybe_download_media (SFT)
      local path                                   ──► unchanged
      "https://host/clip?id=1"                     ──► cache_dir/sha256(url).<ext>, downloaded once, atomic rename
                                                       after the size matches Content-Length;
                                                       <ext> from the url path if it is a media extension,
                                                       else from Content-Type (download.php -> .png)

What each test pins:
  resolve           absolute paths and urls pass through, other schemes fail, relative paths anchor at a
                    symlinked jsonl's own directory
  truncated body    a body shorter than Content-Length leaves no cache entry
  real http         a loopback download.php serving a PNG is cached as .png and replays from the cache after
                    the server stops
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

import io
from email.message import Message
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest

from miles.utils.media import maybe_download_media, resolve_media_uri


def test_resolve_media_uri(tmp_path):
    # dataset/train.jsonl -> manifests/train.jsonl: "clips/a.mp4" lives next to the link, not its target.
    (tmp_path / "manifests").mkdir()
    (tmp_path / "dataset").mkdir()
    (tmp_path / "manifests" / "train.jsonl").write_text("")
    dataset = tmp_path / "dataset" / "train.jsonl"
    dataset.symlink_to(tmp_path / "manifests" / "train.jsonl")
    assert resolve_media_uri("clips/a.mp4", str(dataset)) == str(tmp_path / "dataset" / "clips" / "a.mp4")
    assert resolve_media_uri("/data/a.mp4", str(dataset)) == "/data/a.mp4"
    assert resolve_media_uri("https://media.test/clip.mp4", str(dataset)) == "https://media.test/clip.mp4"
    with pytest.raises(ValueError, match="Unsupported media URI scheme"):
        resolve_media_uri("s3://bucket/a.mp4", str(dataset))


def test_truncated_download_leaves_no_cache_entry(tmp_path, monkeypatch):
    import miles.utils.media as media

    response = io.BytesIO(b"13 bytes only")
    response.headers = Message()
    response.headers["Content-Length"] = "100"
    monkeypatch.setattr(media, "urlopen", lambda *args, **kwargs: response)
    with pytest.raises(ValueError, match="13 of its 100 declared bytes"):
        maybe_download_media("https://media.test/clip.mp4", tmp_path)
    assert not list(tmp_path.iterdir())


def test_real_http_download_replays_from_cache_after_server_stops(tmp_path):
    requests = []
    payload = b"a real HTTP media response"

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/download.php?id=1"
    try:
        local = maybe_download_media(url, tmp_path)
        assert Path(local).read_bytes() == payload
        assert local.endswith(".png")
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert maybe_download_media(url, tmp_path) == local
    assert requests == ["/download.php?id=1"]
