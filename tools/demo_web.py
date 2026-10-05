# ruff: noqa: E501 - the page below is HTML, CSS and JavaScript in one string; its long lines are markup, not Python
"""Live web demo of the robotics pack (AutonBDA, Module 1): a local page where a
question goes through the real retrieval library on the served index, and the
pack's bound demo questions are run and checked as tools/run_demo.py checks
them. Local by default (127.0.0.1; --host 0.0.0.0 serves other machines, as the
public container does - deploy/README.md); it reads the index and the flows file
and writes no file. Run from the module root:

    python tools/demo_web.py [--host 127.0.0.1] [--port 8765] [--index-dir data/index] [--no-answers]

then open http://127.0.0.1:8765/ . This is a demonstration page, not the S12
HTTP endpoint, and nothing it prints is a record. At most MAX_CONNECTIONS
requests are handled at once, and each must arrive whole within READ_DEADLINE_S. With a Gemini key (GEMINI_API_KEY, else conf.info at the repository
root) it also offers written answers (tools/answer_writer.py): at start it asks
Google once for its model list, and each time the page's button is pressed it
sends the question and the entries returned for it; --no-answers sends nothing.
It refuses to start (exit 2) when the index, the flows file or the
embedding model does not load, when the index does not serve the pack, or when
a bound flow names an entry the index does not hold - tools/run_demo.py's
refusals, so the two cannot disagree.
"""

from __future__ import annotations

import argparse
import json
import re
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT / "src"))

from rag4ds.evalhook.demo import DemoFlowError, check, load_flows, unserved  # noqa: E402
from rag4ds.index.build import IndexBuildError  # noqa: E402
from rag4ds.retrieve.contract import GroundingRequest, GroundingRequestError  # noqa: E402
from rag4ds.retrieve.service import RetrievalError  # noqa: E402

sys.path.insert(0, str(MODULE_ROOT / "tools"))
from answer_writer import MODEL, AnswerError, GeminiWriter, load_key, model_available  # noqa: E402

MAX_QUERY_CHARS = 500
MAX_BODY_BYTES = 10_000
PACK = "robotics"
# AR-rag4ds-060 O1: a request that stalls is dropped after this many seconds;
# at most MAX_CONNECTIONS requests are handled at once (the rest get 503)
REQUEST_TIMEOUT_S = 10
# pending item 2 (AR-rag4ds-060 r2 O10): the timeout above is per read, so a request
# trickled a byte at a time never meets it; the whole request - line, headers and
# body - must arrive within this many seconds, or the connection is cut
READ_DEADLINE_S = 15
MAX_CONNECTIONS = 16
BUSY_REPLIES = 8  # F9: busy replies being sent and drained at once
# AR-rag4ds-060 F4: what is read and discarded of a refused request's body
DRAIN_BYTES = 64 * 1024
DRAIN_SECONDS = 2.0
CONTENT_LENGTH = re.compile(r"[0-9]{1,9}")  # ASCII digits only: no sign, no "_", no spaces
BUSY = (b"HTTP/1.0 503 Service Unavailable\r\nContent-Type: text/plain; charset=utf-8\r\n"
        b"Content-Length: 21\r\nConnection: close\r\n\r\nThe server is busy.\r\n")


class DemoStartError(RuntimeError):
    """The demo cannot serve what it promises; it refuses to start."""


class Demo:
    """The retriever and the flows, loaded and checked once at start. The
    embedder and the pack directory are injectable (tests)."""

    def __init__(self, index_dir: Path, embedder=None, pack_dir: Path | None = None):
        from rag4ds.index.build import SentenceTransformerEmbedder, load_index
        from rag4ds.retrieve.service import Retriever

        self.index = load_index(Path(index_dir))
        m = self.index.manifest
        # AR-rag4ds-060 F5: tools/run_demo.py's two refusals, applied before serving
        if PACK not in m["pack_versions"]:
            raise DemoStartError(f"the index at {index_dir} does not serve pack {PACK!r}")
        _, self.flows = load_flows(Path(pack_dir) if pack_dir else MODULE_ROOT / "packs" / PACK)
        held = {row["id"]: row["pack"] for row in self.index.rows}
        missing = unserved(self.flows, set(held))
        if missing:
            raise DemoStartError(f"the flows name entry ids the index at {index_dir} does not "
                                 f"hold - {'; '.join(missing)}")
        self.pack_of = held
        pin = m["embedding_model"]
        emb = embedder or SentenceTransformerEmbedder(pin["id"], pin["revision"])
        # AR-rag4ds-060 F2: the model loads lazily, on the first embed; one embed
        # now makes a pin that cannot load stop the start, not answer every
        # question with ERROR. Any failure is reported by its type and message.
        try:
            emb.embed(["model load check"])
        except Exception as exc:  # re-raised as the start refusal; nothing is swallowed
            raise DemoStartError(f"the embedding model {pin['id']} at revision "
                                 f"{pin['revision']} does not load - {type(exc).__name__}: "
                                 f"{exc}") from exc
        self.retriever = Retriever(self.index, emb)
        self.writer = None  # set by main(): a GeminiWriter, or None with the reason
        self.writer_note = "not set up"

    def about(self) -> dict:
        m = self.index.manifest
        return {"pack": PACK, "pack_version": m["pack_versions"][PACK],
                "index_version": m["index_version"], "entries": m["entry_count"],
                "model": m["embedding_model"]["id"].split("/")[-1],
                "min_relevance": m["min_relevance"], "answers": self.writer is not None,
                "answers_note": self.writer_note,
                "answer_model": MODEL if self.writer is not None else None}

    def ask(self, query) -> dict:
        if not isinstance(query, str) or not query.strip():
            return {"error": "Type a question first."}
        if len(query) > MAX_QUERY_CHARS:
            return {"error": f"Keep the question to at most {MAX_QUERY_CHARS} characters."}
        try:
            req = GroundingRequest(query=query.strip(), domain=PACK, top_k=5)
        except GroundingRequestError as exc:
            return {"error": str(exc)}
        return self.retriever.ground(req).to_dict()

    def answer(self, query) -> dict:
        """A written answer from the entries the library returns for the question, or
        an error saying why there is none; Google is asked only when there is grounding."""
        if self.writer is None:
            return {"error": f"Written answers are off: {self.writer_note}."}
        resp = self.ask(query)
        if "error" in resp:
            return resp
        if resp["status"] != "OK":
            return {"error": "The library found no grounding, so no answer is written."}
        try:
            return self.writer.write(query.strip(), resp["entries"])
        except AnswerError as exc:
            return {"error": f"No written answer: {exc}"}

    def run_flows(self) -> list[dict]:
        out = []
        for f in self.flows:
            resp = self.retriever.ground(f.grounding_request()).to_dict()
            res = check(f, resp)
            e = f.expected
            out.append({
                "id": f.id, "query": f.query, "verdict": res.verdict, "status": resp["status"],
                "reasons": list(res.reasons), "entries": resp.get("entries", []),
                "expected": None if e is None else {
                    "status": e.status, "present": list(e.present), "absent": list(e.absent),
                    "bound_by": e.bound_by, "bound_on": e.bound_on, "record": e.record,
                    # AR-rag4ds-059 r3 O14(c): whether passing needs any entry of the pack itself
                    "requires_pack_entry": any(self.pack_of.get(x) == PACK for x in e.present)},
            })
        return out


PAGE = """<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\">
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
<title>AutonBDA Module 1 - Robotics Knowledge Pack Demonstration</title>
<style>
body{margin:0;background:#ffffff;color:#1a1a1a;font:15px/1.55 \"Segoe UI\",Arial,Helvetica,sans-serif}
.page{max-width:980px;margin:0 auto;padding:24px 20px 48px}
header{border-bottom:2px solid #1a1a1a;padding-bottom:10px;margin-bottom:16px}
h1{font-size:22px;font-weight:600;margin:0 0 4px}
.sub{margin:0;color:#444}
h2{font-size:17px;font-weight:600;margin:30px 0 6px}
p{max-width:78ch;margin:8px 0}
form{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0 6px}
input[type=text]{flex:1 1 320px;min-width:0;font:inherit;padding:6px 8px;border:1px solid #8a8a8a;border-radius:2px}
button{font:inherit;padding:6px 14px;border:1px solid #6b6b6b;border-radius:2px;background:#f0f0f0;color:#1a1a1a;cursor:pointer}
button:hover{background:#e4e4e4}
button:disabled{color:#888;cursor:default}
input:focus-visible,button:focus-visible,a:focus-visible{outline:2px solid #1f4e79;outline-offset:1px}
.examples{font-size:14px;color:#444}
.examples a{color:#1f4e79;margin-right:14px}
.status{font-size:14px;color:#444;margin:8px 0}
.tablewrap{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:14px;margin-top:4px}
th,td{border:1px solid #c8c8c8;padding:6px 8px;text-align:left;vertical-align:top}
th{background:#f3f3f3;font-weight:600}
td.num{text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums}
td.id{white-space:nowrap}
.src{font-size:12.5px;color:#555}
.pass{color:#176b2c;font-weight:600}
.fail{color:#a31515;font-weight:600}
.small{font-size:13px;color:#555}
.answer{border-left:3px solid #c8c8c8;padding:2px 0 2px 12px;margin:12px 0}
.answer p{margin:6px 0}
footer{margin-top:36px;border-top:1px solid #c8c8c8;padding-top:10px;font-size:13px;color:#555}
</style>
</head>
<body>
<div class=\"page\">
<header>
  <h1>AutonBDA &ndash; Module 1: RAG for Data Science</h1>
  <p class=\"sub\" id=\"sub\">Robotics knowledge pack &ndash; retrieval demonstration</p>
</header>

<p>Module 1 of the AutonBDA project is a knowledge framework for data science. Its knowledge base
has three layers &ndash; general concepts, domain knowledge and task patterns &ndash; and every entry is
reviewed and approved by an expert before it can be used. This page queries the robotics knowledge pack.
For each question the library returns up to five approved entries whose relevance score is at or
above the cut-off, each with its source, or reports that it has no grounding for the question.</p>

<h2>Ask a question</h2>
<form id=\"askf\"><input type=\"text\" id=\"q\" maxlength=\"500\" autocomplete=\"off\" aria-label=\"Question\"
  placeholder=\"Type a question about robot reliability or failure analysis\"><button id=\"askb\" type=\"submit\" disabled>Search</button></form>
<div class=\"examples\" id=\"ex\">Examples:</div>
<div class=\"status\" id=\"st\">Connecting to the library...</div>
<div class=\"tablewrap\" id=\"res\"></div>
<div class=\"answer\" id=\"ansbox\" hidden>
<p><button id=\"ansb\" type=\"button\">Write an answer from these entries</button> <span class=\"status\" id=\"anst\"></span></p>
<div id=\"ans\"></div>
<p class=\"small\" id=\"ansnote\"></p>
</div>
<p class=\"small\" id=\"ansoff\"></p>

<h2>Demonstration questions</h2>
<p>Fixed questions with requirements set in advance: entries that must be returned, entries that
must not be returned, or no grounding at all. Each requirement was proposed beforehand and accepted
by the robotics validator. Each run checks the library's current answers against those requirements.</p>
<p><button id=\"runb\" type=\"button\" disabled>Run the demonstration questions</button> <span class=\"status\" id=\"sum\"></span></p>
<div class=\"tablewrap\" id=\"flows\"></div>
<p class=\"small\" id=\"margin\"></p>

<footer id=\"foot\">AutonBDA project, Module 1 (RAG for Data Science).</footer>
</div>
<script>
const $ = (id) => document.getElementById(id);
let cutoff = null;  // the library's own cut-off, read from /api/about before anything is enabled
let answers = false;  // written answers, as /api/about reports them
let lastQ = null;  // the question the shown entries answer
const esc = (s) => String(s).replace(/[&<>\"]/g, (c) => ({\"&\": \"&amp;\", \"<\": \"&lt;\", \">\": \"&gt;\", '\"': \"&quot;\"}[c]));
const layerName = (e) => e.layer === \"domain\" ? \"robotics\" : (e.layer === \"task\" ? \"task pattern\" : \"general concept\");
const srcText = (s) => s.startsWith(\"corpus:\") ? \"catalogued document \" + s.slice(s.indexOf(\"/\") + 1) : s;

function resultTable(entries) {
  const rows = entries.map((e, i) => `<tr><td class=\"num\">${i + 1}</td><td class=\"id\">${esc(e.id)}<br><span class=\"src\">version ${e.version}</span></td>
    <td>${layerName(e)}</td><td class=\"num\">${e.relevance_score.toFixed(3)}</td>
    <td>${esc(e.content)}<br><span class=\"src\">Source: ${esc(srcText(e.source))}</span></td></tr>`).join(\"\");
  return `<table><thead><tr><th>#</th><th>Entry</th><th>Layer</th><th>Score</th><th>Content and source</th></tr></thead><tbody>${rows}</tbody></table>`;
}

async function post(url, body) {
  const r = await fetch(url, {method: \"POST\", headers: {\"Content-Type\": \"application/json\"}, body: JSON.stringify(body || {})});
  return r.json();
}

async function ask(q) {
  if (cutoff === null) return;
  $(\"askb\").disabled = true;
  $(\"st\").textContent = \"Searching...\";
  try {
    const d = await post(\"/api/ask\", {query: q});
    $(\"ansbox\").hidden = true; lastQ = null;
    if (d.error) { $(\"st\").textContent = d.error; $(\"res\").innerHTML = \"\"; return; }
    if (d.status === \"OK\") {
      $(\"st\").textContent = `${d.entries.length} ${d.entries.length === 1 ? \"entry\" : \"entries\"} at or above the cut-off of ${cutoff} for: \"${q}\"`;
      $(\"res\").innerHTML = resultTable(d.entries);
      lastQ = q;
      $(\"ans\").innerHTML = \"\"; $(\"anst\").textContent = \"\";
      $(\"ansbox\").hidden = !answers;
    } else if (d.status === \"NO_GROUNDING\") {
      $(\"st\").textContent = `No grounding for: \"${q}\". No entry reaches the cut-off of ${cutoff}; the library says so instead of guessing.`;
      $(\"res\").innerHTML = \"\";
    } else {
      $(\"st\").textContent = \"The library reported an error: \" + ((d.meta && d.meta.error) || \"unknown\");
      $(\"res\").innerHTML = \"\";
    }
  } catch (err) {
    $(\"st\").textContent = \"The demonstration server did not answer. Is it still running?\";
  } finally {
    $(\"askb\").disabled = false;
  }
}

async function writeAnswer() {
  if (!answers || lastQ === null) return;
  $(\"ansb\").disabled = true;
  $(\"anst\").textContent = \"Writing...\";
  $(\"ans\").innerHTML = \"\";
  try {
    const d = await post(\"/api/answer\", {query: lastQ});
    if (d.error) { $(\"anst\").textContent = d.error; return; }
    if (!d.answerable) { $(\"anst\").textContent = \"The entries found do not answer this question, so no answer is written.\"; return; }
    $(\"anst\").textContent = \"\";
    $(\"ans\").innerHTML = d.sentences.map((s) => `<p>${esc(s.text)} <span class=\"src\">[${s.cites.map(esc).join(\", \")}]</span></p>`).join(\"\");
  } catch (err) {
    $(\"anst\").textContent = \"The demonstration server did not answer. Is it still running?\";
  } finally {
    $(\"ansb\").disabled = false;
  }
}

function requirement(x) {
  if (!x) return \"Not yet bound.\";
  let s = `Status ${esc(x.status)}`;
  if (x.present.length) s += `<br>Must return: ${x.present.map(esc).join(\", \")}`;
  if (x.absent.length) s += `<br>Must not return: ${x.absent.map(esc).join(\", \")}`;
  if (x.present.length && !x.requires_pack_entry) s += `<br><span class=\"src\">No robotics entry is required here; the required ${x.present.length === 1 ? \"entry is\" : \"entries are\"} shared.</span>`;
  return s + `<br><span class=\"src\">Proposed in advance and accepted by ${esc(x.bound_by)}, ${esc(x.bound_on)}; record: ${esc(x.record)}</span>`;
}

function returned(f) {
  if (!f.entries.length) return esc(f.status);
  return f.entries.map((e) => {
    let tag = \"\";
    if (f.expected && f.expected.present.includes(e.id)) tag = ' <span class=\"src\">required</span>';
    if (f.expected && f.expected.absent.includes(e.id)) tag = ' <span class=\"fail\">must not be returned</span>';
    return `${e.relevance_score.toFixed(3)}&nbsp; ${esc(e.id)}${tag}`;
  }).join(\"<br>\");
}

function marginNote(fl) {
  let low = null;
  fl.forEach((f, i) => {
    if (!f.expected) return;
    f.entries.forEach((e) => {
      if (f.expected.present.includes(e.id) && (low === null || e.relevance_score < low.score)) {
        low = {id: e.id, score: e.relevance_score, n: i + 1};
      }
    });
  });
  return low ? `Closest to the cut-off among the required entries: ${esc(low.id)} in question ${low.n}, score ${low.score.toFixed(3)} against a cut-off of ${cutoff}.` : \"\";
}

async function runFlows() {
  if (cutoff === null) return;
  $(\"runb\").disabled = true;
  $(\"sum\").textContent = \"Running...\";
  try {
    const fl = await post(\"/api/flows\");
    const n = fl.filter((f) => f.verdict === \"PASS\").length;
    $(\"sum\").textContent = `${n} of ${fl.length} pass.`;
    const rows = fl.map((f, i) => `<tr><td class=\"num\">${i + 1}</td><td>${esc(f.query)}</td><td>${requirement(f.expected)}</td>
      <td>${returned(f)}</td><td class=\"${f.verdict === \"PASS\" ? \"pass\" : \"fail\"}\">${esc(f.verdict)}${f.reasons.length ? `<br><span class=\"src\">${f.reasons.map(esc).join(\"; \")}</span>` : \"\"}</td></tr>`).join(\"\");
    $(\"flows\").innerHTML = `<table><colgroup><col style=\"width:3%\"><col style=\"width:22%\"><col style=\"width:33%\"><col style=\"width:34%\"><col style=\"width:8%\"></colgroup><thead><tr><th>#</th><th>Question</th><th>Requirement</th><th>Returned (score, entry)</th><th>Result</th></tr></thead><tbody>${rows}</tbody></table>`;
    $(\"margin\").innerHTML = marginNote(fl);
  } catch (err) {
    $(\"sum\").textContent = \"The demonstration server did not answer. Is it still running?\";
  } finally {
    $(\"runb\").disabled = false;
  }
}

const examples = [\"which torque patterns precede actuator failure in rotary joints\",
  \"what data does failure prediction need from a robot arm\",
  \"how do harmonic drives behave under load\",
  \"what are the signs of a bearing fault\"];
$(\"ex\").innerHTML = \"Examples: \" + examples.map((q) => `<a href=\"#\" data-q=\"${esc(q)}\">${esc(q)}</a>`).join(\"\");
$(\"ex\").addEventListener(\"click\", (ev) => {
  const a = ev.target.closest(\"a[data-q]\");
  if (!a) return;
  ev.preventDefault();
  $(\"q\").value = a.dataset.q;
  ask(a.dataset.q);
});
$(\"askf\").addEventListener(\"submit\", (ev) => { ev.preventDefault(); ask($(\"q\").value); });
$(\"runb\").addEventListener(\"click\", runFlows);
$(\"ansb\").addEventListener(\"click\", writeAnswer);
fetch(\"/api/about\").then((r) => r.json()).then((a) => {
  cutoff = a.min_relevance;
  answers = a.answers === true;
  $(\"ansnote\").textContent = answers ? `Written by Google Gemini (${a.answer_model}), which is asked to use only the entries shown above. Every sentence and every line must end with the ids, in square brackets, of the entries it rests on: an answer with one that does not, or with an id that is not one of these entries, is not shown. Only those ids are checked - not whether a sentence is true to its entries, nor any source the text names in other words - and words not split by a stop or a line break count as one sentence. Pressing the button sends your question and those entries to Google.` : \"\";
  $(\"ansoff\").textContent = answers ? \"\" : `Written answers are off: ${a.answers_note}.`;
  $(\"st\").textContent = \"\";
  $(\"askb\").disabled = false;
  $(\"runb\").disabled = false;
  $(\"sub\").textContent = `Robotics knowledge pack, version ${a.pack_version} \u2013 retrieval demonstration`;
  $(\"foot\").textContent = `AutonBDA project, Module 1 (RAG for Data Science). Knowledge index ${a.index_version} (${a.entries} entries); embedding model ${a.model}; relevance cut-off ${a.min_relevance}. Nothing typed on this page is stored.`;
  runFlows();
}).catch(() => {
  $(\"sub\").textContent = \"The demonstration server is not answering.\";
  $(\"st\").textContent = \"The library's cut-off is unknown, so searching is off until the server answers.\";
});
</script>
</body>
</html>
"""


def _half_close_and_drain(conn, what: str) -> None:
    """After a reply sent before the request was read (AR-rag4ds-060 F4, F9): close
    the write side, then read and discard the rest of the request - at most
    DRAIN_BYTES, for at most DRAIN_SECONDS - until the client closes, so nothing
    is unread when the socket closes and Windows does not reset the connection
    before the client has the reply. Failures are reported on stderr."""
    try:
        conn.shutdown(socket.SHUT_WR)
    except OSError as exc:
        sys.stderr.write(f"demo_web: {what}: half-close failed: {exc!r}\n")
        return
    deadline, left = time.monotonic() + DRAIN_SECONDS, DRAIN_BYTES
    while left > 0:
        wait = deadline - time.monotonic()
        if wait <= 0:
            break
        conn.settimeout(wait)
        try:
            chunk = conn.recv(min(4096, left))
        except TimeoutError:
            break  # the client keeps the connection open: the bound ends the wait
        except OSError as exc:
            sys.stderr.write(f"demo_web: {what}: draining stopped: {exc!r}\n")
            break
        if not chunk:
            break
        left -= len(chunk)


def make_handler(demo: Demo, timeout: float = REQUEST_TIMEOUT_S,
                 read_deadline: float = READ_DEADLINE_S):
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self._read_lock = threading.Lock()
            self._read_state = "reading"
            self._watch = threading.Timer(read_deadline, self._cut_slow_request)
            self._watch.daemon = True
            self._watch.start()

        def _cut_slow_request(self) -> None:
            """The request did not arrive within read_deadline: the connection is cut,
            and that is reported on stderr."""
            with self._read_lock:
                if self._read_state != "reading":
                    return
                self._read_state = "cut"
            sys.stderr.write(f"demo_web: {self.client_address} - the request did not arrive "
                             f"within {read_deadline:g} s; connection cut\n")
            try:
                self.connection.shutdown(socket.SHUT_RDWR)
            except OSError as exc:
                sys.stderr.write(f"demo_web: cutting {self.client_address} failed: {exc!r}\n")

        def _request_read(self) -> bool:
            """The whole request is in: the watch stops. False when it was cut first."""
            with self._read_lock:
                if self._read_state == "cut":
                    return False
                self._read_state = "read"
            self._watch.cancel()
            return True

        def finish(self):
            self._watch.cancel()
            super().finish()

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj).encode("utf-8"), "application/json; charset=utf-8")

        def _refuse(self, message: str) -> None:
            """AR-rag4ds-060 F4: a request refused before its body is read. Closing
            a socket with unread input makes Windows reset the connection, and the
            reset can reach the client before the 400 does. So the 400 is sent,
            the write side is closed, and the rest of the request is read and
            discarded (at most DRAIN_BYTES, for at most DRAIN_SECONDS) until the
            client closes - then nothing is unread when the socket closes."""
            if not self._request_read():  # its headers are in; the drain has its own bound
                return
            self._json({"error": message}, 400)
            self.wfile.flush()
            _half_close_and_drain(self.connection, "refused request")

        def do_GET(self):  # noqa: N802 - the http.server hook's name
            if not self._request_read():
                return
            if self.path in ("/", "/index.html"):
                self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            elif self.path == "/api/about":
                self._json(demo.about())
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self):  # noqa: N802
            # AR-rag4ds-060 O1: one Content-Length, of ASCII digits only; no chunked bodies
            if self.headers.get("Transfer-Encoding") is not None:
                return self._refuse("a chunked request body is not accepted; send a Content-Length")
            lengths = self.headers.get_all("Content-Length") or []
            if len(lengths) > 1:
                return self._refuse("the request has more than one Content-Length")
            # spaces or tabs around a header value are HTTP's own syntax (RFC 9110,
            # 5.5), not part of the value; inside it, only digits
            raw = lengths[0].strip(" \t") if lengths else "0"
            if not CONTENT_LENGTH.fullmatch(raw):
                return self._refuse("the request's Content-Length is not a byte count")
            n = int(raw)
            if n > MAX_BODY_BYTES:
                return self._refuse(f"the request body must be 0 to {MAX_BODY_BYTES} bytes")
            try:
                data = self.rfile.read(n)
            except OSError:
                if not self._request_read():  # cut at the read deadline: nothing to answer
                    return
                raise
            if not self._request_read():
                return
            try:
                body = json.loads(data or b"{}")
            except (ValueError, RecursionError):  # O1: a deeply nested body is not JSON we read
                return self._json({"error": "the request was not JSON"}, 400)
            if self.path == "/api/ask":
                self._json(demo.ask(body.get("query") if isinstance(body, dict) else None))
            elif self.path == "/api/answer":
                self._json(demo.answer(body.get("query") if isinstance(body, dict) else None))
            elif self.path == "/api/flows":
                self._json(demo.run_flows())
            else:
                self._json({"error": "not found"}, 404)

        def log_message(self, fmt, *args):  # quiet console; the page is the output
            pass

        def log_error(self, fmt, *args):  # errors are not quiet (charter section 4)
            sys.stderr.write(f"demo_web: {self.address_string()} - {fmt % args}\n")

    Handler.timeout = timeout  # AR-rag4ds-060 O1: a stalled request is dropped
    return Handler


class DemoServer(ThreadingHTTPServer):
    """AR-rag4ds-060 O1: at most max_connections requests are handled at once;
    a request beyond that gets 503 - delivered as a 400 is (F9) - instead of a
    handler thread."""

    daemon_threads = True

    def __init__(self, address, handler, max_connections: int = MAX_CONNECTIONS):
        self._slots = threading.BoundedSemaphore(max_connections)
        self._busy = threading.BoundedSemaphore(BUSY_REPLIES)
        super().__init__(address, handler)

    def _busy_reply(self, request, client_address):
        try:
            try:
                request.sendall(BUSY)
            except OSError as exc:
                sys.stderr.write(f"demo_web: the busy reply to {client_address} failed: {exc!r}\n")
            else:
                _half_close_and_drain(request, f"busy reply to {client_address}")
            request.close()
        finally:
            self._busy.release()

    def process_request(self, request, client_address):
        if not self._slots.acquire(blocking=False):
            # AR-rag4ds-060 F9: the 503 is sent, then the request is drained, as a
            # 400 is - from a short thread, so the accept loop is never held. At most
            # BUSY_REPLIES such threads at once; beyond that the connection is closed
            # with no reply, and that is reported on stderr.
            if self._busy.acquire(blocking=False):
                threading.Thread(target=self._busy_reply, args=(request, client_address),
                                 daemon=True).start()
            else:
                sys.stderr.write(f"demo_web: {client_address} closed with no reply - the "
                                 f"server and its busy replies are all in use\n")
                self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._slots.release()


def setup_writer(demo: Demo, no_answers: bool, writer=None) -> None:
    """Written answers on, or off with the reason, which the page shows."""
    if writer is not None:  # tests
        demo.writer, demo.writer_note = writer, "on"
        return
    if no_answers:
        demo.writer_note = "turned off when the server was started (--no-answers)"
        return
    try:
        key = load_key()
        if key is None:
            demo.writer_note = "no key (GEMINI_API_KEY or conf.info)"
        elif not model_available(key):
            demo.writer_note = f"the model {MODEL} is not offered to this key"
        else:
            demo.writer, demo.writer_note = GeminiWriter(key), "on"
    except AnswerError as exc:  # shown on the page and printed; never carries the key
        demo.writer_note = str(exc)


def main(argv: list[str] | None = None, *, embedder=None, pack_dir: Path | None = None,
         writer=None) -> int:
    """embedder, pack_dir and writer are for tests only; the command line never sets them."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--host", default="127.0.0.1",
                    help="address to serve on; 0.0.0.0 serves other machines (the container)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--index-dir", default=str(MODULE_ROOT / "data" / "index"))
    ap.add_argument("--no-answers", action="store_true",
                    help="no written answers: nothing is sent to Google")
    args = ap.parse_args(argv)
    try:
        demo = Demo(Path(args.index_dir), embedder=embedder, pack_dir=pack_dir)
        server = DemoServer((args.host, args.port), make_handler(demo))
    except (IndexBuildError, DemoFlowError, DemoStartError, RetrievalError, OSError) as exc:
        print(f"demo_web: cannot start - {exc}", file=sys.stderr)
        return 2
    print(f"demo_web: AutonBDA Module 1 robotics demo at http://{args.host}:{server.server_address[1]}/ "
          f"(index {demo.about()['index_version']}); Ctrl+C to stop", flush=True)
    setup_writer(demo, args.no_answers, writer)
    print(f"demo_web: written answers {'on (' + MODEL + ')' if demo.writer else 'off - ' + demo.writer_note}",
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
