"""Written answers (task 5; tools/answer_writer.py and the demonstration page's panel): an
answer is shown only when every sentence cites entries it was given; nothing is asked of
the provider without grounding; the key never leaves the server; the hourly bound holds.
The provider is a FAKE request function - no network, no key - except where a test
says otherwise (none does)."""

import hashlib
import importlib.util
import json
import threading
from pathlib import Path

import numpy as np
import pytest

from rag4ds.index.build import build_index

MODULE_ROOT = Path(__file__).resolve().parents[1]
TS = "2026-10-05 16:00:00 JST"
FAKE_KEY = "FAKE-gemini-key-for-tests-not-a-real-key"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"{name}_t5",
                                                  MODULE_ROOT / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


aw = _load("answer_writer")
ENTRIES = [{"id": "harmonic-drive-torque", "content": "c1", "source": "corpus:robotics/a"},
           {"id": "bearing-outer-race-fault", "content": "c2", "source": "corpus:robotics/b"}]


def FAKE_request(text, calls=None):  # noqa: N802 - a clearly named test double
    def request(method, path, key, body=None):
        if calls is not None:
            calls.append((method, path, key, body))
        return {"candidates": [{"content": {"parts": [{"text": text}]}}]}
    return request


def test_an_answer_whose_every_sentence_cites_given_entries_is_shown():
    w = aw.GeminiWriter(FAKE_KEY, request=FAKE_request(
        "Harmonic drives give high torque [harmonic-drive-torque]. Bearings wear "
        "[bearing-outer-race-fault, harmonic-drive-torque]."))
    out = w.write("q", ENTRIES)
    assert out["answerable"] and out["model"] == aw.MODEL
    assert out["sentences"] == [
        {"text": "Harmonic drives give high torque.", "cites": ["harmonic-drive-torque"]},
        {"text": "Bearings wear.", "cites": ["bearing-outer-race-fault", "harmonic-drive-torque"]}]


@pytest.mark.parametrize("text, why", [
    ("Harmonic drives give high torque [harmonic-drive-torque]. Bearings wear.", "cites no entry"),
    # AR-rag4ds-062 F1: an uncited sentence however it begins, and text after the last citation
    ("Drives carry torque [harmonic-drive-torque]. they never fail.", "cites no entry"),
    ("Drives carry torque [harmonic-drive-torque].They never fail.", "cites no entry"),
    ("- Drives carry torque [harmonic-drive-torque]\n- They never fail.", "cites no entry"),
    ("Drives carry torque [harmonic-drive-torque]. urScript runs everywhere.", "cites no entry"),
    ("Drives carry torque [harmonic-drive-torque] and never fail", "cites no entry"),
    # one sentence per citation
    ("Drives carry torque. They never fail [harmonic-drive-torque].", "more than one sentence"),
    # AR-rag4ds-062 F4: an uncited line, item or sentence BEFORE a cited one, however it ends
    ("- They never fail\n- Drives carry torque [harmonic-drive-torque]", "cites no entry"),
    ("Drives never fail\nThey carry torque [harmonic-drive-torque].", "cites no entry"),
    ("**Drives never fail.** They carry torque [harmonic-drive-torque].", "more than one sentence"),
    ('"Drives never fail." They carry torque [harmonic-drive-torque].', "more than one sentence"),
    ("(Drives never fail.) They carry torque [harmonic-drive-torque].", "more than one sentence"),
    ("Drives never fail\u2026 They carry torque [harmonic-drive-torque].",
     "more than one sentence"),
    ("Drives never fail\u3002They carry torque [harmonic-drive-torque].", "more than one sentence"),
    ("Drives never fail\uff01They carry torque [harmonic-drive-torque].", "more than one sentence"),
    ("Drives never fail.\u200bThey carry torque [harmonic-drive-torque].",
     "more than one sentence"),
    ("Drives carry torque [harmonic-drive-torque]. **They never fail.** Races wear "
     "[bearing-outer-race-fault].", "more than one sentence"),
    ("Drives carry torque [harmonic-drive-torque].\n\nThey never fail\n\nRaces wear "
     "[bearing-outer-race-fault].", "cites no entry"),
    ("- Drives carry torque [harmonic-drive-torque]\n- They never fail\n- Races wear "
     "[bearing-outer-race-fault]", "cites no entry"),
    ("- **Drives never fail.** They carry torque [harmonic-drive-torque]",
     "more than one sentence"),
    ("## Answer\nDrives carry torque [harmonic-drive-torque].", "cites no entry"),
    ("Drives carry torque\u2028They never fail [harmonic-drive-torque].", "cites no entry"),
    # the shape report r2 saw live (L3)
    ("**Harmonic drives handle variable loading with stability, precision, and high torque.** "
     "Under load, they provide high torque and minimal backlash in a compact unit "
     "[harmonic-drive-torque].", "more than one sentence"),
    ("NOT ANSWERABLE [harmonic-drive-torque]", "NOT ANSWERABLE with a citation"),
    # AR-rag4ds-062 F2: a bracket that is not a list of entry ids
    ("Drives wear out [Wikipedia] [harmonic-drive-torque].", "not a list of entry ids"),
    ("Drives wear out [see harmonic-drive-torque].", "not a list of entry ids"),
    ("- [harmonic-drive-torque].", "no sentence before it"),
    ("[harmonic-drive-torque].", "cites no entry"),
    ("Drives fail [made-up-entry].", "ids that were not given"),
    ("", "empty"),
    (" ".join(f"Sentence {i} [harmonic-drive-torque]." for i in range(8)), "sentences"),
])
def test_an_answer_that_does_not_cite_as_asked_is_refused(text, why):
    w = aw.GeminiWriter(FAKE_KEY, request=FAKE_request(text))
    with pytest.raises(aw.AnswerError, match=why):
        w.write("q", ENTRIES)


@pytest.mark.parametrize("text, sentences", [
    ("Drives carry torque [harmonic-drive-torque]",
     [("Drives carry torque.", ["harmonic-drive-torque"])]),
    ("- Drives carry torque [harmonic-drive-torque]\n- Races wear [bearing-outer-race-fault]",
     [("Drives carry torque.", ["harmonic-drive-torque"]),
      ("Races wear.", ["bearing-outer-race-fault"])]),
    ("Drives carry torque [harmonic-drive-torque] [bearing-outer-race-fault]!",
     [("Drives carry torque!", ["harmonic-drive-torque", "bearing-outer-race-fault"])]),
    ("Drives carry torque. [harmonic-drive-torque]",
     [("Drives carry torque.", ["harmonic-drive-torque"])]),
    # r2 O10, O11: a numbered list, markdown bullets and bold, paragraphs, each citing
    ("1. Drives carry torque [harmonic-drive-torque]\n2. Races wear [bearing-outer-race-fault]",
     [("Drives carry torque.", ["harmonic-drive-torque"]),
      ("Races wear.", ["bearing-outer-race-fault"])]),
    ("* **Drives** carry torque [harmonic-drive-torque]",
     [("Drives carry torque.", ["harmonic-drive-torque"])]),
    ("Drives carry torque [harmonic-drive-torque].\n\nRaces wear [bearing-outer-race-fault].",
     [("Drives carry torque.", ["harmonic-drive-torque"]),
      ("Races wear.", ["bearing-outer-race-fault"])]),
    ("Drives carry 1.5 times the torque [harmonic-drive-torque].",
     [("Drives carry 1.5 times the torque.", ["harmonic-drive-torque"])]),
    # declared (record 85): words no stop separates are one sentence and share the citation
    # that closes them
    ("Drives never fail; they carry torque [harmonic-drive-torque].",
     [("Drives never fail; they carry torque.", ["harmonic-drive-torque"])]),
    ("Drives never fail - they carry torque [harmonic-drive-torque].",
     [("Drives never fail - they carry torque.", ["harmonic-drive-torque"])]),
])
def test_the_cited_forms_that_are_shown(text, sentences):
    out = aw.check_answer(text, {e["id"] for e in ENTRIES})
    assert [(s["text"], s["cites"]) for s in out] == sentences


@pytest.mark.parametrize("reply", ["NOT ANSWERABLE", "NOT ANSWERABLE.", " not answerable. ",
                                   "**NOT ANSWERABLE**", "NOT ANSWERABLE!", "NOT  ANSWERABLE"])
def test_not_answerable_with_or_without_a_full_stop(reply):
    w = aw.GeminiWriter(FAKE_KEY, request=FAKE_request(reply))
    assert w.write("q", ENTRIES)["answerable"] is False


def test_not_answerable_is_passed_on_and_the_prompt_carries_only_the_entries():
    calls = []
    w = aw.GeminiWriter(FAKE_KEY, request=FAKE_request(aw.NOT_ANSWERABLE, calls))
    assert w.write("which bolts loosen", ENTRIES) == {"answerable": False, "sentences": [],
                                                      "model": aw.MODEL}
    (method, path, key, body), = calls
    assert method == "POST" and path == f"/v1beta/models/{aw.MODEL}:generateContent"
    prompt = body["contents"][0]["parts"][0]["text"]
    assert "Question: which bolts loosen" in prompt and "[harmonic-drive-torque] c1" in prompt
    assert "Use no other knowledge" in prompt and key == FAKE_KEY and FAKE_KEY not in prompt


def test_the_hourly_bound_holds():
    now = [0.0]
    w = aw.GeminiWriter(FAKE_KEY, request=FAKE_request(aw.NOT_ANSWERABLE), clock=lambda: now[0])
    for _ in range(aw.ANSWERS_PER_HOUR):
        w.write("q", ENTRIES)
    with pytest.raises(aw.AnswerError, match="written answers an hour"):
        w.write("q", ENTRIES)
    now[0] = 3601.0
    assert w.write("q", ENTRIES)["answerable"] is False


def test_the_key_comes_from_the_environment_or_the_file_and_never_from_a_bad_file(tmp_path):
    f = tmp_path / "conf.info"
    assert aw.load_key({}, f) is None
    f.write_text(f"GEMINI_API_KEY={FAKE_KEY}\n", encoding="utf-8")
    assert aw.load_key({}, f) == FAKE_KEY
    assert aw.load_key({"GEMINI_API_KEY": "from-env"}, f) == "from-env"
    f.write_text("something else\n", encoding="utf-8")
    with pytest.raises(aw.AnswerError, match="not one line GEMINI_API_KEY=<key>") as e:
        aw.load_key({}, f)
    assert FAKE_KEY not in str(e.value)


def test_with_no_key_file_the_key_comes_from_the_environment_only():
    """The public container (/app) has no repository root above it, so no conf.info:
    the module still loads, and the key is GEMINI_API_KEY or nothing."""
    assert aw.load_key({}, None) is None
    assert aw.load_key({"GEMINI_API_KEY": FAKE_KEY}, None) == FAKE_KEY


def test_a_provider_error_never_carries_the_key(monkeypatch):
    class Resp:
        status = 400

        def read(self):
            return f'{{"error": "bad key {FAKE_KEY}"}}'.encode()

    class Conn:
        def __init__(self, *a, **k):
            pass

        def request(self, *a, **k):
            pass

        def getresponse(self):
            return Resp()

        def close(self):
            pass

    monkeypatch.setattr(aw, "_IPv4HTTPSConnection", Conn)
    with pytest.raises(aw.AnswerError) as e:
        aw._request("GET", "/v1beta/models", FAKE_KEY)
    assert "HTTP 400" in str(e.value) and FAKE_KEY not in str(e.value) and "<key>" in str(e.value)


def test_a_reply_that_is_not_json_is_an_answer_error(monkeypatch):
    class Resp:
        status = 200

        def read(self):
            return b"<html>not json</html>"

    class Conn:
        def __init__(self, *a, **k):
            pass

        def request(self, *a, **k):
            pass

        def getresponse(self):
            return Resp()

        def close(self):
            pass

    monkeypatch.setattr(aw, "_IPv4HTTPSConnection", Conn)
    with pytest.raises(aw.AnswerError, match="reply was not JSON"):
        aw._request("GET", "/v1beta/models", FAKE_KEY)


def test_the_connection_asks_for_ipv4_addresses_only(monkeypatch):
    seen = {}

    def gai(host, port, family=0, *a, **k):
        seen["family"] = family
        raise OSError("stop here")

    monkeypatch.setattr(aw.socket, "getaddrinfo", gai)
    with pytest.raises(aw.AnswerError, match="Google did not answer"):
        aw._request("GET", "/v1beta/models", FAKE_KEY)
    assert seen["family"] == aw.socket.AF_INET


# --- the page and its endpoint ---------------------------------------------------------

class FAKE_BagOfWordsEmbedder:  # noqa: N801 - a clearly named test double
    model_id, revision, dim = "FAKE-bag-of-words", "test-double", 4096

    def embed(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in t.lower().replace("-", " ").split():
                out[i, int(hashlib.sha256(w.encode()).hexdigest(), 16) % self.dim] += 1.0
            if not out[i].any():
                out[i, 0] = 1.0
        return out


@pytest.fixture(scope="module")
def page(tmp_path_factory):
    web = _load("demo_web")
    out = tmp_path_factory.mktemp("answers") / "index"
    build_index(MODULE_ROOT, out, FAKE_BagOfWordsEmbedder(), {"robotics": "0.0.0"}, 0.05,
                "0000000", TS)
    demo = web.Demo(out, embedder=FAKE_BagOfWordsEmbedder())
    return web, demo


def _http(web, demo, method, path, obj=None):
    import http.client
    server = web.DemoServer(("127.0.0.1", 0), web.make_handler(demo))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=30)
        raw = None if obj is None else json.dumps(obj).encode()
        conn.request(method, path, body=raw, headers={} if raw is None else
                     {"Content-Type": "application/json", "Content-Length": str(len(raw))})
        r = conn.getresponse()
        data = r.read()
        conn.close()
        return r.status, data
    finally:
        server.shutdown()
        server.server_close()


def test_with_no_writer_the_page_says_answers_are_off_and_nothing_is_asked(page):
    web, demo = page
    web.setup_writer(demo, no_answers=True)
    st, data = _http(web, demo, "GET", "/api/about")
    a = json.loads(data)
    assert a["answers"] is False and "--no-answers" in a["answers_note"]
    assert a["answer_model"] is None
    st, data = _http(web, demo, "POST", "/api/answer", {"query": "harmonic drive torque"})
    assert json.loads(data)["error"].startswith("Written answers are off")


def FAKE_citing_request(calls):  # noqa: N802 - a test double: cites the first entry given
    def request(method, path, key, body=None):
        calls.append((method, path, key, body))
        prompt = body["contents"][0]["parts"][0]["text"]
        first = prompt.split("Entries:\n", 1)[1].split("]", 1)[0].lstrip("[")
        return {"candidates": [{"content": {"parts": [{"text": f"It answers that [{first}]."}]}}]}
    return request


def test_the_endpoint_answers_from_the_entries_the_library_returned(page):
    web, demo = page
    calls = []
    web.setup_writer(demo, False, web.GeminiWriter(FAKE_KEY, request=FAKE_citing_request(calls)))
    st, data = _http(web, demo, "GET", "/api/about")
    assert json.loads(data)["answers"] is True and json.loads(data)["answer_model"] == web.MODEL
    query = "harmonic drive torque under load"
    first = demo.ask(query)["entries"][0]["id"]
    st, data = _http(web, demo, "POST", "/api/answer", {"query": query})
    d = json.loads(data)
    assert st == 200 and d["answerable"] and d["sentences"] == [
        {"text": "It answers that.", "cites": [first]}]
    assert len(calls) == 1 and FAKE_KEY.encode() not in data
    # no grounding (no word in common with any entry): the provider is not asked
    assert demo.ask("zzqx vvkw")["status"] == "NO_GROUNDING"
    st, data = _http(web, demo, "POST", "/api/answer", {"query": "zzqx vvkw"})
    assert json.loads(data)["error"] == "The library found no grounding, so no answer is written."
    assert len(calls) == 1
    # a refused answer is reported, not shown; a reply that is not JSON is reported too
    web.setup_writer(demo, False, web.GeminiWriter(FAKE_KEY, request=FAKE_request("Uncited.")))
    st, data = _http(web, demo, "POST", "/api/answer", {"query": query})
    assert json.loads(data)["error"].startswith("No written answer: text that cites no entry")


def test_the_page_carries_the_panel_and_says_where_questions_go(page):
    web, demo = page
    st, data = _http(web, demo, "GET", "/")
    html = data.decode("utf-8")
    assert 'id="ansb"' in html and 'id="ansbox" hidden' in html
    assert "Pressing the button sends your question and those entries to Google." in html
    assert ("Every sentence and every line must end with the ids, in square brackets, of the "
            "entries it rests on") in html
    assert "Only those ids are checked - not whether a sentence is true to its entries" in html
    assert "words not split by a stop or a line break count as one sentence." in html
    assert FAKE_KEY not in html
