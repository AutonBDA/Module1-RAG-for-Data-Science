"""The live local web demo (tools/demo_web.py): its page names AutonBDA Module 1 and
is a plain white page that loads nothing from the network; its API validates what
it is sent; it refuses to start (exit 2) on everything tools/run_demo.py refuses
and on a model that does not load; and its demonstration questions give the demo
runner's own verdicts. Driven over real HTTP on a free port, against an index built
from the live gated tree with a FAKE embedder - the served index and the real
model are never touched."""

import codecs
import dataclasses
import hashlib
import http.client
import importlib.util
import io
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from rag4ds.evalhook.demo import FLOWS_FILE, check, load_flows
from rag4ds.index.build import build_index, load_index

MODULE_ROOT = Path(__file__).resolve().parents[1]
TS = "2026-10-05 02:00:00 JST"
H1 = "AutonBDA &ndash; Module 1: RAG for Data Science"


class FAKE_BagOfWordsEmbedder:  # noqa: N801 - a clearly named test double (charter section 8)
    model_id = "FAKE-bag-of-words"
    revision = "test-double"
    dim = 4096

    def embed(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in t.lower().replace("-", " ").split():
                out[i, int(hashlib.sha256(w.encode()).hexdigest(), 16) % self.dim] += 1.0
            if not out[i].any():
                out[i, 0] = 1.0
        return out


class FAKE_UnloadableEmbedder(FAKE_BagOfWordsEmbedder):  # noqa: N801 - test double
    def embed(self, texts):
        raise OSError("FAKE: the model is not in the cache and the Hub is offline")


def _tool():
    spec = importlib.util.spec_from_file_location("demo_web_tool",
                                                  MODULE_ROOT / "tools" / "demo_web.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _serve(tool, demo, **kw):
    timeout = kw.pop("timeout", tool.REQUEST_TIMEOUT_S)
    handler = kw.pop("handler", None) or tool.make_handler(demo, timeout=timeout)
    server = tool.DemoServer(("127.0.0.1", 0), handler, **kw)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    tool = _tool()
    out = tmp_path_factory.mktemp("demo-web") / "index"
    build_index(MODULE_ROOT, out, FAKE_BagOfWordsEmbedder(), {"robotics": "0.0.0"}, 0.05,
                "0000000", TS)
    demo = tool.Demo(out, embedder=FAKE_BagOfWordsEmbedder())
    server = _serve(tool, demo)
    yield tool, demo, server.server_address[1], out
    server.shutdown()
    server.server_close()


def _req(port, method, path, body=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    conn.request(method, path, body=body, headers=headers or {})
    r = conn.getresponse()
    data = r.read()
    conn.close()
    return r.status, r.getheader("Content-Type"), data


def _post(port, path, obj):
    raw = json.dumps(obj).encode()
    st, _, data = _req(port, "POST", path, raw, {"Content-Type": "application/json",
                                                "Content-Length": str(len(raw))})
    return st, json.loads(data)


def _raw(port, request: bytes, timeout=10) -> bytes:
    """Send exact bytes; read until the server closes."""
    with socket.create_connection(("127.0.0.1", port), timeout=timeout) as s:
        s.sendall(request)
        chunks = []
        while True:
            c = s.recv(65536)
            if not c:
                return b"".join(chunks)
            chunks.append(c)


def _luminance(hex_colour):
    h = hex_colour.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    rgb = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def test_the_page_names_the_project_and_is_plain_white_and_self_contained(served):
    _, _, port, _ = served
    st, ctype, data = _req(port, "GET", "/")
    page = data.decode("utf-8")
    assert st == 200 and ctype.startswith("text/html")
    # AR-rag4ds-060 F3: the heading's own text, not words found anywhere on the page
    assert re.findall(r"<h1>(.*?)</h1>", page, re.S) == [H1]
    # white: the body is #ffffff, and EVERY background the page sets is light
    # (a hex colour of relative luminance >= 0.7 - the darkest the page uses is
    # the button hover #e4e4e4, 0.78; a dark panel such as #121212 is 0.006 and
    # mid-grey #808080 0.22; rgb(), a gradient or an image fails); no dark theme
    assert re.search(r"body\{[^}]*background:#ffffff", page)
    backgrounds = re.findall(r"background(?:-color)?\s*:\s*([^;}\"]+)", page)
    assert len(backgrounds) >= 4
    for b in backgrounds:
        assert re.fullmatch(r"#[0-9a-fA-F]{3}|#[0-9a-fA-F]{6}", b.strip()), b
        assert _luminance(b.strip()) >= 0.7, b
    for dark in ("prefers-color-scheme", "color-scheme", "filter:", "invert(", "mix-blend"):
        assert dark not in page, dark
    # nothing fetched from the network, in any form - protocol-relative included
    assert not re.search(r"(?i)(?:https?:)?//[a-z0-9-]+(?:\.[a-z0-9-]+)+", page)
    for ref in ("@import", "url(", "<link", " src=", "<iframe", "<img"):
        assert ref not in page.lower(), ref
    assert set(re.findall(r'href="([^"]*)"', page)) <= {"#"}
    # no AI coding tool's mark (2026-10-05): the marks are kept ROT13-encoded so this file
    # names no tool; Google Gemini is named on purpose - it writes the optional answers
    # and the page says so (tools/answer_writer.py)
    for mark in ("pynhqr", "naguebcvp", "trarengrq jvgu", "pbcvybg", "bcranv"):
        assert codecs.decode(mark, "rot13") not in page.lower(), mark
    # AR-rag4ds-060 F6: no cut-off written into the page; the script reads the
    # library's, and both buttons stay off until it has
    assert not re.search(r"cutoff\s*=\s*-?[0-9.]", page)
    assert "let cutoff = null;" in page and "cutoff = a.min_relevance;" in page
    assert 'id="askb" type="submit" disabled>' in page
    assert 'id="runb" type="button" disabled>' in page


def test_about_reports_the_index_it_serves(served):
    _, demo, port, _ = served
    st, _, data = _req(port, "GET", "/api/about")
    a = json.loads(data)
    assert st == 200 and a["pack"] == "robotics" and a["pack_version"] == "0.0.0"
    assert a["index_version"] == demo.index.manifest["index_version"] and a["min_relevance"] == 0.05


def test_ask_refuses_an_empty_or_overlong_question_and_answers_a_real_one(served):
    tool, _, port, _ = served
    assert _post(port, "/api/ask", {})[1] == {"error": "Type a question first."}
    assert _post(port, "/api/ask", {"query": "   "})[1] == {"error": "Type a question first."}
    long = "x" * (tool.MAX_QUERY_CHARS + 1)
    assert "at most" in _post(port, "/api/ask", {"query": long})[1]["error"]
    st, d = _post(port, "/api/ask", {"query": "harmonic drive torque under load"})
    assert st == 200 and d["status"] == "OK" and 1 <= len(d["entries"]) <= 5
    fields = {"id", "version", "layer", "content", "relevance_score", "source"}
    assert fields <= set(d["entries"][0])


def test_the_demonstration_questions_give_the_runners_own_verdicts(served):
    tool, demo, port, index_dir = served
    st, fl = _post(port, "/api/flows", {})
    _, flows = load_flows(MODULE_ROOT / "packs" / "robotics")
    assert st == 200 and [f["id"] for f in fl] == [f.id for f in flows]
    pack_of = {r["id"]: r["pack"] for r in load_index(index_dir).rows}
    for got, f in zip(fl, flows, strict=True):
        want = check(f, demo.retriever.ground(f.grounding_request()).to_dict())
        assert got["verdict"] == want.verdict and got["reasons"] == list(want.reasons)
        assert got["expected"]["record"] == f.expected.record
        assert got["expected"]["bound_by"] == f.expected.bound_by
        assert got["expected"]["requires_pack_entry"] == any(
            pack_of[x] == "robotics" for x in f.expected.present)
    # AR-rag4ds-060 F5: and the runner itself, on the same index, says the same
    spec = importlib.util.spec_from_file_location("run_demo_tool",
                                                  MODULE_ROOT / "tools" / "run_demo.py")
    run_demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_demo)
    out = io.StringIO()
    rc = run_demo.execute(MODULE_ROOT, "robotics", index_dir, embedder=FAKE_BagOfWordsEmbedder(),
                          stream=out)
    lines = [ln.split(":")[0].split() for ln in out.getvalue().splitlines()
             if ln.split() and ln.split()[0] in ("PASS", "FAIL", "UNBOUND")]
    assert [(v, i) for v, i in lines] == [(f["verdict"], f["id"]) for f in fl]
    assert rc == (0 if all(f["verdict"] == "PASS" for f in fl) else 1)


@pytest.mark.parametrize("length", ["abc", "-1", "999999", "10001", "+2", "0_2", "1e3"])
def test_a_bad_content_length_is_refused_with_400_every_time(served, length):
    """AR-rag4ds-060 F4: the refusal reaches the client every time - it was
    reset before it on Windows in about one run in fifteen; O1: a sign, an
    underscore or an exponent is not a byte count."""
    _, _, port, _ = served
    for _ in range(25):
        st, _, data = _req(port, "POST", "/api/ask", b"{}", {"Content-Length": length})
        assert st == 400 and "error" in json.loads(data)


def test_spaces_around_the_content_length_are_http_syntax_not_part_of_it(served):
    """RFC 9110 5.5: optional whitespace around a field value is not part of the value."""
    _, _, port, _ = served
    got = _raw(port, b"POST /api/ask HTTP/1.0\r\nContent-Length:  2 \t\r\n\r\n{}")
    assert got.startswith(b"HTTP/1.0 200") and b"Type a question first." in got


def test_two_content_lengths_and_a_chunked_body_are_refused(served):
    _, _, port, _ = served
    two = _raw(port, b"POST /api/ask HTTP/1.1\r\nHost: x\r\nContent-Length: 2\r\n"
                     b"Content-Length: 5\r\n\r\n{}")
    assert two.startswith(b"HTTP/1.0 400") and b"more than one Content-Length" in two
    chunked = _raw(port, b"POST /api/ask HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n"
                         b"\r\n2\r\n{}\r\n0\r\n\r\n")
    assert chunked.startswith(b"HTTP/1.0 400") and b"chunked" in chunked


def test_the_body_size_boundary_is_exact(served):
    """AR-rag4ds-060 F3: MAX_BODY_BYTES bytes are read and answered; one more is refused."""
    tool, _, port, _ = served
    q = b'{"query": "harmonic drive torque under load"}'
    full = q + b" " * (tool.MAX_BODY_BYTES - len(q))
    assert len(full) == tool.MAX_BODY_BYTES == 10_000
    st, _, data = _req(port, "POST", "/api/ask", full, {"Content-Length": str(len(full))})
    assert st == 200 and json.loads(data)["status"] == "OK"
    over = full + b" "
    st, _, data = _req(port, "POST", "/api/ask", over, {"Content-Length": str(len(over))})
    assert st == 400 and json.loads(data)["error"] == "the request body must be 0 to 10000 bytes"


def test_a_body_that_is_not_json_and_an_unknown_path_are_refused(served):
    _, _, port, _ = served
    st, _, data = _req(port, "POST", "/api/ask", b"not json", {"Content-Length": "8"})
    assert st == 400 and json.loads(data)["error"] == "the request was not JSON"
    deep = b"[" * 10_000  # O1: nested past the decoder's recursion limit
    st, _, data = _req(port, "POST", "/api/ask", deep, {"Content-Length": str(len(deep))})
    assert st == 400 and json.loads(data)["error"] == "the request was not JSON"
    assert _req(port, "GET", "/nowhere")[0] == 404
    assert _post(port, "/api/nowhere", {})[0] == 404


def test_a_stalled_request_is_dropped_after_the_timeout(served, capfd):
    """AR-rag4ds-060 O1: a body shorter than its Content-Length no longer holds a
    thread until the client leaves."""
    tool, demo, _, _ = served
    server = _serve(tool, demo, timeout=1)
    try:
        t0 = time.monotonic()
        got = _raw(server.server_address[1],
                   b"POST /api/ask HTTP/1.0\r\nContent-Length: 10\r\n\r\n{}", timeout=10)
        assert got == b"" and time.monotonic() - t0 < 5
    finally:
        server.shutdown()
        server.server_close()
    assert "timed out" in capfd.readouterr().err


def _trickle(port, head: bytes, drip: bytes, every: float, limit: float) -> tuple[bytes, float]:
    """Send head, then one byte of drip every `every` seconds, until the server closes
    the connection or `limit` passes; returns what the server sent and when it closed."""
    t0 = time.monotonic()
    got = b""
    with socket.create_connection(("127.0.0.1", port), timeout=limit) as s:
        s.sendall(head)
        s.settimeout(every)
        for i in range(int(limit / every)):
            try:
                s.sendall(drip[i % len(drip):i % len(drip) + 1])
                c = s.recv(65536)
                if not c:
                    break
                got += c
            except TimeoutError:
                continue
            except OSError:  # reset or aborted by the cut
                break
    return got, time.monotonic() - t0


@pytest.mark.parametrize("head, drip", [
    # pending item 2: a body trickled a byte at a time, each byte inside the per-read timeout
    (b"POST /api/ask HTTP/1.0\r\nContent-Length: 9000\r\n\r\n", b'{"query": "a"}'),
    # and headers trickled the same way, before any body
    (b"GET /api/about HTTP/1.0\r\n", b"X-Slow: aaaaaaaa\r\n"),
])
def test_a_request_trickled_slowly_is_cut_at_the_read_deadline(served, capfd, head, drip):
    tool, demo, _, _ = served
    server = _serve(tool, demo, handler=tool.make_handler(demo, timeout=10, read_deadline=1.5))
    try:
        got, took = _trickle(server.server_address[1], head, drip, every=0.25, limit=12)
        assert got == b"" and 1.0 < took < 6, (got[:60], took)
    finally:
        server.shutdown()
        server.server_close()
    assert "the request did not arrive within 1.5 s; connection cut" in capfd.readouterr().err


def test_a_prompt_request_is_not_cut_by_the_read_deadline(served):
    """The deadline covers the arrival of the request, not the time spent answering it."""
    tool, demo, _, _ = served

    class Slow(tool.make_handler(demo, timeout=10, read_deadline=0.5)):
        def do_GET(self):  # noqa: N802
            if self._request_read():
                time.sleep(1.5)  # answering takes longer than the deadline
                self._json({"ok": True})

    server = _serve(tool, demo, handler=Slow)
    try:
        got = _raw(server.server_address[1], b"GET /api/about HTTP/1.0\r\n\r\n")
        assert got.startswith(b"HTTP/1.0 200") and got.endswith(b'{"ok": true}'), got[:80]
    finally:
        server.shutdown()
        server.server_close()


def test_the_stated_bounds_are_the_ones_in_force(served):
    """AR-rag4ds-060 r2 O7: record 74's figures - 16 requests at once, a 10-second
    request timeout - are the defaults the server and handler actually take."""
    tool, demo, _, _ = served
    assert (tool.MAX_CONNECTIONS, tool.REQUEST_TIMEOUT_S, tool.BUSY_REPLIES) == (16, 10, 8)
    assert tool.READ_DEADLINE_S == 15
    assert tool.make_handler(demo).timeout == 10
    server = tool.DemoServer(("127.0.0.1", 0), tool.make_handler(demo))
    try:
        assert server._slots._value == 16 and server._busy._value == 8
    finally:
        server.server_close()


def test_requests_beyond_the_connection_bound_get_503(served):
    """AR-rag4ds-060 O1 and r2 F9: the number of requests handled at once is
    bounded, and a request beyond the bound gets the 503 even when it has sent
    its whole request, as a browser does - every time, with no retry."""
    tool, demo, _, _ = served
    entered = threading.Event()

    class Held(tool.make_handler(demo, timeout=10)):
        def setup(self):
            super().setup()
            entered.set()

    server = _serve(tool, demo, handler=Held, max_connections=1)
    port = server.server_address[1]
    try:
        held = socket.create_connection(("127.0.0.1", port), timeout=10)
        assert entered.wait(10)
        for request in [b"GET /api/about HTTP/1.1\r\nHost: x\r\nUser-Agent: test\r\n\r\n"] * 30 + [
                b"POST /api/ask HTTP/1.1\r\nHost: x\r\nContent-Length: 2\r\n\r\n{}", b""]:
            busy = _raw(port, request)
            assert busy.startswith(b"HTTP/1.0 503"), busy[:60]
            assert busy.endswith(b"The server is busy.\r\n")
        held.close()
        got = b""
        for _ in range(100):  # the slot comes back once the held request ends
            got = _raw(port, b"GET /api/about HTTP/1.0\r\n\r\n")
            if got.startswith(b"HTTP/1.0 200"):
                break
            assert got.startswith(b"HTTP/1.0 503"), got[:60]  # busy, never a reset
            time.sleep(0.05)
        assert got.startswith(b"HTTP/1.0 200")
    finally:
        server.shutdown()
        server.server_close()


def _refuses(tool, monkeypatch, argv, **kw):
    """main() must refuse; if it starts the server instead, the test fails at
    once rather than serving forever."""
    def started(self, *a, **k):
        raise AssertionError("main() started the server instead of refusing")

    monkeypatch.setattr(tool.DemoServer, "serve_forever", started)
    return tool.main(argv, **kw)


def test_main_refuses_an_index_that_does_not_load(tmp_path, capsys, monkeypatch):
    tool = _tool()
    missing = str(tmp_path / "missing")
    assert _refuses(tool, monkeypatch, ["--index-dir", missing, "--port", "0"]) == 2
    assert "cannot start" in capsys.readouterr().err


def _pack_copy(tmp_path, flows_text):
    pack = tmp_path / "robotics"
    (pack / FLOWS_FILE).parent.mkdir(parents=True)
    (pack / FLOWS_FILE).write_text(flows_text, encoding="utf-8")
    return pack


def test_main_refuses_a_flows_file_that_does_not_load(served, tmp_path, capsys, monkeypatch):
    """AR-rag4ds-060 F3: the flows arm of main()'s refusal."""
    tool, _, _, index_dir = served
    pack = _pack_copy(tmp_path, "schema: 1\npack: robotics\nflows: []\n")
    assert _refuses(tool, monkeypatch, ["--index-dir", str(index_dir), "--port", "0"],
                    embedder=FAKE_BagOfWordsEmbedder(), pack_dir=pack) == 2
    assert "flows must be a non-empty list" in capsys.readouterr().err


def test_main_refuses_flows_that_name_an_entry_the_index_does_not_hold(served, tmp_path, capsys,
                                                                      monkeypatch):
    """AR-rag4ds-060 F5: run_demo's refusal - such a constraint can never fire or never be met."""
    tool, _, _, index_dir = served
    text = (MODULE_ROOT / "packs" / "robotics" / FLOWS_FILE).read_text(encoding="utf-8")
    assert "- association-rule-analysis\n" in text
    pack = _pack_copy(tmp_path, text.replace("- association-rule-analysis\n",
                                             "- association-rule-analysis-retired\n", 1))
    assert _refuses(tool, monkeypatch, ["--index-dir", str(index_dir), "--port", "0"],
                    embedder=FAKE_BagOfWordsEmbedder(), pack_dir=pack) == 2
    err = capsys.readouterr().err
    assert "does not hold" in err and "association-rule-analysis-retired" in err


def test_demo_refuses_an_index_that_does_not_serve_the_pack(served, monkeypatch):
    tool, demo, _, index_dir = served
    import rag4ds.index.build as build

    other = dataclasses.replace(demo.index, manifest={**demo.index.manifest, "pack_versions": {}})
    monkeypatch.setattr(build, "load_index", lambda _p: other)
    with pytest.raises(tool.DemoStartError, match="does not serve pack 'robotics'"):
        tool.Demo(index_dir, embedder=FAKE_BagOfWordsEmbedder())


def test_main_refuses_a_model_that_does_not_load(served, capsys, monkeypatch):
    """AR-rag4ds-060 F2: the model is loaded at start, so a pin that cannot load
    stops the start instead of answering every question with ERROR."""
    tool, _, _, index_dir = served
    assert _refuses(tool, monkeypatch, ["--index-dir", str(index_dir), "--port", "0"],
                    embedder=FAKE_UnloadableEmbedder()) == 2
    err = capsys.readouterr().err
    assert "does not load - OSError: FAKE: the model is not in the cache" in err


def test_the_real_embedder_with_no_model_in_reach_stops_the_start(served, tmp_path):
    """AR-rag4ds-060 F2, the auditor's own probe as a test: the tool as run from the
    command line, with an empty model cache and the Hub offline, exits 2 at start."""
    _, _, _, index_dir = served
    empty = tmp_path / "hf"
    env = {**os.environ, "HF_HOME": str(empty), "HF_HUB_CACHE": str(empty / "hub"),
           "SENTENCE_TRANSFORMERS_HOME": str(empty / "st"), "HF_HUB_OFFLINE": "1",
           "TRANSFORMERS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    try:  # a tool that starts serving instead of refusing is killed and fails the test
        p = subprocess.run([sys.executable, "-B", str(MODULE_ROOT / "tools" / "demo_web.py"),
                            "--index-dir", str(index_dir), "--port", "0"],
                           capture_output=True, text=True, env=env, timeout=120)
    except subprocess.TimeoutExpired as exc:
        pytest.fail(f"the tool did not exit within 120 s: {exc}")
    assert p.returncode == 2, (p.returncode, p.stdout[-500:], p.stderr[-500:])
    assert "cannot start" in p.stderr and "does not load" in p.stderr
    assert "robotics demo at" not in p.stdout
