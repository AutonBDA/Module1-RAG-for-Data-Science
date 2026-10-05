"""Written answers for the demonstration page (task 5 of the PI's five of 2026-10-05; his
choice C of 01:09:43 JST: Google Gemini, the key in conf.info).

The library retrieves; Google Gemini is asked to answer the question from the retrieved
entries alone, ending every sentence with the ids of the entries it rests on. What is
enforced is the form (AR-rag4ds-062 F1, F2, F4): every line of the reply is a run of
sentences, each closed by a bracket of ids of entries it was given, and nothing else;
otherwise the reply is refused, and the page says so. A sentence ends at a line end or at
any stop - a full stop, ! or ?, an ellipsis, an ideographic or full-width stop, any other
punctuation but , ; : ' " % & # @ / and a full stop between two digits - so an
abbreviation such as "e.g." inside a sentence is refused too; markdown asterisks are
dropped first. Words that no stop separates (clauses joined by a dash or a semicolon) are
one sentence and share the citation that closes them. Whether each sentence is true to
the entries it cites is NOT checked, nor is a source the text names in other words, such
as "(Wikipedia)". Nothing is asked when the library found no grounding.

- The key: the environment variable GEMINI_API_KEY, else the file conf.info at the
  repository root (one line GEMINI_API_KEY=<key>, ignored by git). It is sent to Google
  as a header and nowhere else: never printed, logged, returned or put in the page.
- The model: the one named here, used only if Google's own model list offers it.
- The network: requests go over IPv4 (this machine's IPv6 route to Google stalls - a
  probe of 2026-10-05), with a timeout on every request.
- A bound: at most ANSWERS_PER_HOUR written answers per server process.
The provider is called over its own REST API with the standard library; no third-party
client library is used.
"""

from __future__ import annotations

import http.client
import json
import os
import re
import socket
import ssl
import threading
import time
import unicodedata
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
# conf.info at the repository root; a copy of the module with no repository around it
# (the public container, /app) has none and reads GEMINI_API_KEY only
KEY_FILE = MODULE_ROOT.parents[1] / "conf.info" if len(MODULE_ROOT.parents) > 1 else None
HOST = "generativelanguage.googleapis.com"
MODEL = "gemini-3.8-flash"  # 2.5 is listed but closed to new keys (404, 2026-10-05)
TIMEOUT_S = 30
ANSWERS_PER_HOUR = 60
MAX_SENTENCES = 4
NOT_ANSWERABLE = "NOT ANSWERABLE"
ID = re.compile(r"[a-z0-9][a-z0-9-]*")
# one segment: a sentence (no brackets in it), then one or more bracket groups, then an
# optional end mark - each line of the reply must be a run of these and nothing else
# (AR-rag4ds-062 F1, F4)
SEGMENT = re.compile(r"\s*(?P<text>[^\[\]]+?)\s*(?P<ids>(?:\[[^\[\]]*\]\s*)+)(?P<end>[.!?])?\s*")
# what may begin a line before its first sentence: a list marker or a heading mark
LIST_MARK = re.compile("^(?:[-+\u2022\u2023\u2043\u25e6]|#{1,6}|[0-9]{1,3}[.)])(?:\\s+|$)")
# the "other punctuation" (Unicode Po) a sentence may hold inside it; any other ends it
INNER = set(",;:'\"%&#@/")
CLOSING = ("Po", "Pe", "Pf", "Cf")
SHOWN_END = re.compile("[.!?\u2026\u3002\uff01\uff1f][\\\"'\u201d\u2019)\\]]*$")


def _holds_a_stop(sentence: str) -> bool:
    """Whether a stop stands inside the sentence rather than only at its end (F4)."""
    core = sentence  # what may close it: stops, closing brackets and quotes, format marks
    while core and (core[-1].isspace() or unicodedata.category(core[-1]) in CLOSING):
        core = core[:-1]
    for i, ch in enumerate(core):
        if unicodedata.category(ch) != "Po" or ch in INNER:
            continue
        if ch == "." and 0 < i < len(core) - 1 and core[i - 1].isdigit() and core[i + 1].isdigit():
            continue
        return True
    return False


class AnswerError(RuntimeError):
    """A written answer could not be produced or was refused; the message says why and
    never carries the key."""


def load_key(env=None, key_file: Path | None = KEY_FILE) -> str | None:
    """The key from GEMINI_API_KEY, else conf.info; None when neither has one."""
    env = os.environ if env is None else env
    key = (env.get("GEMINI_API_KEY") or "").strip()
    if key:
        return key
    if key_file is None or not Path(key_file).is_file():
        return None
    raw = Path(key_file).read_text(encoding="utf-8")
    m = re.match(r"\s*GEMINI_API_KEY\s*[=:]\s*(\S+)\s*$", raw)
    if not m:
        raise AnswerError(f"{Path(key_file).name} is not one line GEMINI_API_KEY=<key>")
    return m.group(1)


class _IPv4HTTPSConnection(http.client.HTTPSConnection):
    """HTTPS over IPv4 only: getaddrinfo is asked for AF_INET addresses."""

    def connect(self):
        addr = socket.getaddrinfo(self.host, self.port, socket.AF_INET, socket.SOCK_STREAM)[0][4]
        sock = socket.create_connection(addr, self.timeout)
        self.sock = ssl.create_default_context().wrap_socket(sock, server_hostname=self.host)


def _request(method: str, path: str, key: str, body: dict | None = None) -> dict:
    conn = _IPv4HTTPSConnection(HOST, 443, timeout=TIMEOUT_S)
    try:
        payload = None if body is None else json.dumps(body).encode("utf-8")
        headers = {"x-goog-api-key": key}
        if payload is not None:
            headers["Content-Type"] = "application/json"
        conn.request(method, path, body=payload, headers=headers)
        r = conn.getresponse()
        data = r.read()
    except OSError as exc:  # timeouts and network failures, reported without the key
        raise AnswerError(f"Google did not answer: {type(exc).__name__}") from None
    finally:
        conn.close()
    if r.status != 200:
        detail = data.decode("utf-8", "replace").replace(key, "<key>")[:200]
        raise AnswerError(f"Google answered HTTP {r.status}: {detail}")
    try:
        return json.loads(data)
    except ValueError:
        raise AnswerError("Google's reply was not JSON") from None


def model_available(key: str, request=_request) -> bool:
    """Whether Google's own model list offers MODEL."""
    names = [m.get("name", "") for m in request("GET", "/v1beta/models?pageSize=200", key)
             .get("models", [])]
    return f"models/{MODEL}" in names


def prompt_for(query: str, entries: list[dict]) -> str:
    lines = [f"[{e['id']}] {e['content']} (source: {e['source']})" for e in entries]
    return (
        "Answer the question below using ONLY the entries listed after it. Write at most "
        f"{MAX_SENTENCES} short sentences. End every sentence with the id(s) of the entries it "
        "rests on, in square brackets, for example [harmonic-drive-torque]. Use no other "
        "knowledge. If the entries do not answer the question, reply exactly: "
        f"{NOT_ANSWERABLE}\n\nQuestion: {query}\n\nEntries:\n" + "\n".join(lines))


def check_answer(text: str, allowed: set[str]) -> list[dict]:
    """Reads each line of the reply as segments, each one sentence closed by its citation
    group(s), and returns [{"text", "cites"}]; raises AnswerError unless
    (AR-rag4ds-062 F1, F2, F4):
    - every line is such segments, with nothing after its last citation group - whatever
      the case of a letter, the spacing or a list marker - so a line, a list item or a
      paragraph with no citation of its own is refused, before a cited one or after it;
    - each segment holds one sentence: no stop inside it (see _holds_a_stop), after the
      markdown asterisks are dropped;
    - every bracket is a list of entry ids, and every id cited is one it was given;
    - no sentence is NOT ANSWERABLE beside a citation."""
    lines = [" ".join(ln.split()) for ln in text.replace("*", "").splitlines()]
    lines = [ln for ln in lines if ln]
    if not lines:
        raise AnswerError("the written answer was empty")
    out = []
    for line in lines:
        pos = 0
        while pos < len(line):
            m = SEGMENT.match(line, pos)
            if m is None:
                rest = line[pos:].strip()
                raise AnswerError(f"text that cites no entry: {rest[:80]!r}")
            sentence = m.group("text").strip()
            if pos == 0:
                sentence = LIST_MARK.sub("", sentence).strip()
            if not re.search(r"[^\W\d_]", sentence):
                raise AnswerError(f"a citation with no sentence before it: "
                                  f"{m.group(0).strip()[:80]!r}")
            if _holds_a_stop(sentence):
                raise AnswerError(f"more than one sentence before one citation: {sentence[:80]!r}")
            groups = re.findall(r"\[([^\[\]]*)\]", m.group("ids"))
            cites = [c.strip() for g in groups for c in g.split(",")]
            if not all(ID.fullmatch(c) for c in cites):
                raise AnswerError(f"a bracket that is not a list of entry ids: "
                                  f"{m.group('ids')[:80]!r}")
            bad = [c for c in cites if c not in allowed]
            if bad:
                raise AnswerError(f"the answer cites ids that were not given: {bad}")
            if sentence.rstrip(".!? ").upper() == NOT_ANSWERABLE:
                raise AnswerError("the reply gave NOT ANSWERABLE with a citation")
            if not SHOWN_END.search(sentence):
                sentence += m.group("end") or "."
            out.append({"text": sentence, "cites": cites})
            pos = m.end()
    if len(out) > MAX_SENTENCES + 2:
        raise AnswerError(f"the answer has {len(out)} sentences; at most {MAX_SENTENCES} "
                          f"were asked")
    return out


class GeminiWriter:
    """Writes one answer per call; request is injectable (tests). Counts calls per hour."""

    def __init__(self, key: str, request=_request, clock=time.monotonic):
        self.key, self.request, self.clock = key, request, clock
        self._calls: list[float] = []
        self._lock = threading.Lock()

    def write(self, query: str, entries: list[dict]) -> dict:
        with self._lock:
            now = self.clock()
            self._calls = [t for t in self._calls if now - t < 3600]
            if len(self._calls) >= ANSWERS_PER_HOUR:
                raise AnswerError(f"the limit of {ANSWERS_PER_HOUR} written answers an hour "
                                  f"is reached")
            self._calls.append(now)
        body = {"contents": [{"parts": [{"text": prompt_for(query, entries)}]}],
                "generationConfig": {"temperature": 0.2, "maxOutputTokens": 600,
                                     "thinkingConfig": {"thinkingBudget": 0}}}
        reply = self.request("POST", f"/v1beta/models/{MODEL}:generateContent", self.key, body)
        try:
            text = "".join(p.get("text", "") for p in reply["candidates"][0]["content"]["parts"])
        except (KeyError, IndexError, TypeError):
            raise AnswerError("Google's reply carried no text") from None
        plain = " ".join(text.replace("*", "").split()).rstrip(".!").strip().upper()
        if plain == NOT_ANSWERABLE:  # r1 O5, r2 O10: a stop, asterisks or spacing around it
            return {"answerable": False, "sentences": [], "model": MODEL}
        return {"answerable": True, "sentences": check_answer(text, {e["id"] for e in entries}),
                "model": MODEL}
