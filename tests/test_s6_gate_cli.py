"""S6 acceptance (plan 10.1): a recorded session over synthetic candidates
proving FUNCTION only - explicitly not a throughput figure. The scripted
sessions exercise approve (id assignment, quotation-bound confirmation),
edit-then-approve, reject, group + mechanical reject-duplicate on the
head's decision, supersession, the conflict flag (validated against the
shown set), mapping-draft approval, character-offset source-span rendering
and its loud unavailability / out-of-range refusal, markup-safe rendering,
review context on every decision, atomic ordering with crash repair that
also closes the head's cluster, refusal without abort, the manifest on
abort, retraction from the journal, the scaling guard, and the recorded
transcript; then the fold compiles exactly what the session approved.
"""

import hashlib
import io
import time

import pytest
from rich.console import Console

from rag4ds.auditlog.journal import Journal
from rag4ds.gate.cli import (
    GateContext,
    GateError,
    GateSession,
    display_provenance_of,
    make_related_finder,
    make_source_resolver,
    suggest_id,
)
from rag4ds.gate.identities import IdentityRegistry
from rag4ds.kb.fold import compile_tree, fold
from rag4ds.staging.store import StagingStore

WHO = "k-naruse"
COMMIT = "23acf2d"
T0 = "2026-08-23 13:10 JST"
T1 = "2026-08-23 13:11 JST"
T2 = "2026-08-23 13:12 JST"
T3 = "2026-08-23 13:13 JST"
T4 = "2026-08-23 13:14 JST"
TEXT = "Intro. Bearing wear dominates actuator failure in rotary joints. Outro."
JP = "\u524d\u66f8\u304d\uff1a\u65e5\u672c\u8a9e\u306e\u5e8f\u6587\u3002"


def clock_factory():
    n = [0]

    def clock():
        n[0] += 1
        return f"2026-08-23 14:{min(n[0], 59):02d}:00 JST"
    return clock


def scripted(answers):
    it = iter(answers)

    def read(prompt):
        try:
            return next(it)
        except StopIteration as exc:
            raise AssertionError(f"script exhausted at prompt {prompt!r}") from exc
    return read


def entry_candidate(content, source_hash, start, end, **o):
    c = {"kind": "entry",
         "proposed": {"type": "concept", "content": content,
                      "source_reference": "corpus:robotics/handbook#s3"},
         "layer": "domain", "pack": "robotics", "source_hash": source_hash,
         "span_start": start, "span_end": end, "extractor": "rule-v1",
         "run_manifest": "runs/extract.jsonl:1"}
    c.update(o)
    return c


def clean_finder(content, layer, pack):
    return []


def setup(tmp_path, with_source=True, related=clean_finder, text=TEXT, **ctx_kw):
    for d in ("journal", "stage", "runs"):
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    doc = tmp_path / "data" / "robotics" / "handbook.txt"
    doc.parent.mkdir(parents=True)
    raw = text.encode("utf-8")
    doc.write_bytes(raw)
    h = hashlib.sha256(raw).hexdigest()
    runs_ingest = Journal(tmp_path / "runs" / "ingest.jsonl")
    runs_ingest.append({"kind": "ingest", "hash": h, "local_path": "robotics/handbook.txt",
                        "pack": "robotics", "schema": 1, "license": "CC-BY", "source": "ptr",
                        "size_bytes": len(raw), "timestamp": "2026-08-23 13:00 JST",
                        "tool": "t", "code_commit": COMMIT, "config": {}})
    resolver = make_source_resolver(runs_ingest, tmp_path / "data") if with_source else None
    ctx_kw.setdefault("identities", IdentityRegistry.of(WHO, "Mapping Team"))  # AR-rag4ds-026 F3
    ctx = GateContext(
        decisions=Journal(tmp_path / "journal" / "decisions.jsonl"),
        staging=StagingStore(tmp_path / "stage" / "candidates.jsonl"),
        runs=Journal(tmp_path / "runs" / "gate-sessions.jsonl"),
        approver=WHO, code_commit=COMMIT, clock=clock_factory(),
        source_resolver=resolver, related_finder=related,
        display_provenance={"index_version": "0123456789abcdef",
                            "embedding_model": {"id": "FAKE", "revision": "t"},
                            "min_relevance": 0.05},
        **ctx_kw,
    )
    return ctx, h, text


def run_session(ctx, answers, **kw):
    buf = io.StringIO()
    console = Console(file=buf, record=True, width=100, force_terminal=False, color_system=None,
                      markup=False, highlight=False)
    s = GateSession(ctx, console, scripted(answers), **kw)
    try:
        summary = s.run()
    finally:
        transcript = console.export_text()
    return summary, transcript


def span_of(text, phrase):
    s = text.index(phrase)
    return s, s + len(phrase)


# --- the recorded function session ---


def test_recorded_session_covers_every_decision_type_and_compiles(tmp_path):
    ctx, h, text = setup(tmp_path)
    st = ctx.staging
    s, e = span_of(text, "Bearing wear dominates actuator failure in rotary joints.")
    c1 = st.file_candidate(entry_candidate("Bearing wear dominates actuator failure.", h, s, e), T0)
    c2 = st.file_candidate(
        entry_candidate("Bearing wear dominate actuator failure (dup).", h, s, e), T1)
    c3 = st.file_candidate(entry_candidate("Torque signature precedes bearing failure.", h, s, e),
                           T2)
    c4 = st.file_candidate(entry_candidate("Unsupported lubrication claim.", h, 0, 6), T3)
    c5 = st.file_candidate({"kind": "mapping-draft", "pack": "robotics",
                            "standard": "cora-ieee-1872", "author": "Mapping Team",
                            "yaml_draft": "standard: cora-ieee-1872\nrows:\n- entry: bearing-wear\n"
                                          "  class: cora:Failure\n  relation: narrower-than\n"
                                          "  note: n\n"}, T4)
    answers = [
        "s",                                        # c1 skipped (stays PENDING)
        "g", c1,                                    # c2 grouped under head c1
        "e", "Torque signature precedes bearing failure in rotary joints.",  # c3 edited
        "y", "torque-signature", "",                # quotation bound, id, no supersession
        "r", "unsupported by the source",           # c4 rejected
        "a",                                        # c5 mapping approved
        "q",
    ]
    summary, transcript = run_session(ctx, answers)
    assert summary.counts["grouped"] == 1 and summary.counts["edit-then-approve"] == 1
    assert summary.counts["reject"] == 1 and summary.counts["approve"] == 1
    assert summary.counts["skipped"] == 1 and summary.aborted is None
    assert st.state_of(c1) == "PENDING" and st.state_of(c2) == "GROUPED"
    assert st.state_of(c3) == "APPROVED" and st.state_of(c4) == "REJECTED"
    assert st.state_of(c5) == "APPROVED"
    assert "Bearing wear dominates actuator failure in rotary joints." in transcript
    assert "quotation bound" in transcript and "session summary" in transcript
    # review context is on every decision
    decs = ctx.decisions.read_all()
    ap = [d for d in decs if d.get("candidate_id") == c3][0]
    assert ap["review"] == {"conflict_display": "checked-clean", "related_shown": [],
                            "conflicts": [], "source_span": "rendered",
                            "quotation_bound_confirmed": True,
                            # AR-rag4ds-026 F5: the record names the index it was decided against
                            "display_provenance": {"index_version": "0123456789abcdef",
                                                   "embedding_model": {"id": "FAKE",
                                                                       "revision": "t"},
                                                   "min_relevance": 0.05},
                            # AR-rag4ds-027 F2: an edit is recorded - who, how much kept
                            "edit": {"editor": WHO, "kept": 1.0, "own": 0.288, "source_run": 21}}
    rj = [d for d in decs if d.get("candidate_id") == c4][0]
    assert rj["review"]["quotation_bound_confirmed"] is False
    # second session: approve the head c1 -> c2 closes mechanically as reject-duplicate
    summary2, transcript2 = run_session(ctx, ["a", "y", "bearing-wear", "", "q"])
    assert summary2.counts["approve"] == 1 and summary2.counts["reject-duplicate"] == 1
    assert st.state_of(c1) == "APPROVED" and st.state_of(c2) == "REJECTED-DUPLICATE"
    decs = ctx.decisions.read_all()
    dup = [d for d in decs if d["type"] == "reject-duplicate"]
    head_dec = [d for d in decs if d.get("candidate_id") == c1][0]
    assert dup[0]["head_decision"] == f"journal/decisions.jsonl:{decs.index(head_dec) + 1}"
    assert dup[0]["candidate_id"] == c2
    assert dup[0]["review"]["conflict_display"] == "not-applicable"  # F13
    assert st.head_decision_of(c1) == dup[0]["head_decision"]
    # the fold compiles exactly the approved entries + the mapping artifact, nothing else
    result = compile_tree(ctx.decisions, tmp_path / "tree")
    assert result.versions == {"torque-signature": 1, "bearing-wear": 1}
    files = sorted(p.relative_to(tmp_path / "tree").as_posix()
                   for p in (tmp_path / "tree").rglob("*") if p.is_file())
    assert files == ["packs/robotics/entries/bearing-wear.json",
                     "packs/robotics/entries/torque-signature.json",
                     "packs/robotics/mappings/cora-ieee-1872.json"]
    # the run manifests (NFR2) name the display's index and model
    runs = ctx.runs.read_all()
    assert [r["kind"] for r in runs] == ["gate-session", "gate-session"]
    assert runs[0]["code_commit"] == COMMIT and runs[0]["approver"] == WHO
    assert runs[0]["config"]["conflict_display"]["index_version"] == "0123456789abcdef"
    assert runs[0]["config"]["allow_unchecked_approval"] is False
    assert runs[0]["counts"]["skipped"] == 1 and runs[1]["counts"]["reject-duplicate"] == 1
    assert runs[0]["counts"]["candidates_source_unavailable"] == 0
    assert runs[0]["counts"]["decisions_without_source"] == 0
    assert runs[0]["aborted"] is None and runs[0]["started"] < runs[0]["ended"]


def test_supersession_and_conflict_flags_validated_against_the_shown_set(tmp_path):
    calls = []

    def finder(content, layer, pack):
        calls.append((content, layer, pack))
        return [{"id": "bearing-wear", "relevance_score": 0.9, "content": "old text"},
                {"id": "misalignment", "relevance_score": 0.4, "content": "other"}]

    ctx, h, text = setup(tmp_path, related=finder)
    st = ctx.staging
    st.file_candidate(entry_candidate("Bearing wear v1.", h, 0, 6), T0)
    st.file_candidate(entry_candidate("Bearing wear v2, broader.", h, 0, 6), T1)
    st.file_candidate(entry_candidate("Lubrication matters.", h, 0, 6), T2)
    answers = [
        "a", "y", "bearing-wear", "", "",                          # c1: new, no conflict flag
        "a", "y", "bearing-wear", "bearing-wear", "",              # c2: supersession v2
        "a", "y", "lubrication", "", "no-such-entry",              # c3: flag not shown -> REFUSED
        "a", "y", "lubrication", "", "bearing-wear, misalignment", # c3 again: two flags
        "q",
    ]
    summary, transcript = run_session(ctx, answers)
    assert summary.counts["approve"] == 3 and summary.counts["refused"] == 1
    assert "REFUSED" in transcript and "not among the shown related entries" in transcript
    decs = ctx.decisions.read_all()
    assert decs[1]["supersedes"] == {"id": "bearing-wear", "version": 1}
    assert decs[1]["entry"]["version"] == 2
    assert decs[2]["conflict_with"] == "bearing-wear"
    assert decs[2]["review"]["conflicts"] == ["bearing-wear", "misalignment"]
    assert decs[2]["review"]["conflict_display"] == "checked-with-results"
    assert decs[2]["review"]["related_shown"] == ["bearing-wear", "misalignment"]
    assert "related approved entries" in transcript
    assert fold(decs, tmp_path / "f").versions == {"bearing-wear": 2, "lubrication": 1}


def test_f1_f2_spans_are_character_offsets_strict_utf8_and_bounds_checked(tmp_path):
    text = JP + TEXT
    ctx, h, text = setup(tmp_path, text=text)
    s, e = span_of(text, "Bearing wear dominates actuator failure in rotary joints.")
    ctx.staging.file_candidate(entry_candidate("x", h, s, e), T0)
    _, transcript = run_session(ctx, ["q"])
    assert "Bearing wear dominates actuator failure in rotary joints." in transcript
    assert "\ufffd" not in transcript and (
        "\u65e5\u672c\u8a9e" not in transcript.split("source span")[1][:200])
    # beyond EOF and overrun both refuse loudly, naming the candidate and the length
    ctx2, h2, text2 = setup(tmp_path / "two")
    ctx2.staging.file_candidate(entry_candidate("x", h2, len(text2) + 5, len(text2) + 9), T0)
    _, t2 = run_session(ctx2, ["q"])
    assert "SOURCE INVALID" in t2 and "outside the document" in t2
    ctx3, h3, text3 = setup(tmp_path / "three")
    ctx3.staging.file_candidate(entry_candidate("x", h3, 10, len(text3) + 1), T0)
    _, t3 = run_session(ctx3, ["q"])
    assert "outside the document" in t3
    # a non-UTF-8 document refuses instead of rendering replacement characters
    ctx4, h4, _ = setup(tmp_path / "four")
    bad = b"\xff\xfe Bearing"
    (tmp_path / "four" / "data" / "robotics" / "handbook.txt").write_bytes(bad)
    ctx4.staging.file_candidate(entry_candidate("x", hashlib.sha256(bad).hexdigest(), 0, 3), T0)
    runs_ingest = Journal(tmp_path / "four" / "runs" / "ingest.jsonl")
    runs_ingest.append({"kind": "ingest", "hash": hashlib.sha256(bad).hexdigest(),
                        "local_path": "robotics/handbook.txt", "pack": "robotics", "schema": 1,
                        "license": "CC-BY", "source": "ptr", "size_bytes": 3,
                        "timestamp": "2026-08-23 13:00 JST", "tool": "t",
                        "code_commit": COMMIT, "config": {}})
    _, t4 = run_session(ctx4, ["q"])
    assert "SOURCE INVALID" in t4 and "not UTF-8" in t4


def test_f4_untrusted_text_is_rendered_verbatim_never_as_markup(tmp_path):
    text = "Intro [b]bold[/b] and [/] and [bold red]red[/bold red] end."
    ctx, h, text = setup(tmp_path, text=text)
    ctx.staging.file_candidate(
        entry_candidate("Content with [bold red]markup[/bold red] inside.", h, 0, len(text)), T0)
    _, transcript = run_session(ctx, ["q"])
    assert "[bold red]markup[/bold red]" in transcript
    assert "[b]bold[/b] and [/] and [bold red]red[/bold red]" in transcript


def test_source_unavailable_is_loud_and_counted_and_hash_mismatch_is_invalid(tmp_path):
    ctx, h, text = setup(tmp_path, with_source=False)
    ctx.staging.file_candidate(entry_candidate("x claim", h, 0, 6), T0)
    summary, transcript = run_session(ctx, ["q"])
    assert "SOURCE UNAVAILABLE" in transcript
    assert summary.counts["candidates_source_unavailable"] == 1
    ctx2, h2, text2 = setup(tmp_path / "two")
    ctx2.staging.file_candidate(entry_candidate("x claim", "f" * 64, 0, 6), T0)
    _, transcript2 = run_session(ctx2, ["q"])
    assert "SOURCE UNAVAILABLE: no local document" in transcript2
    ctx3, h3, text3 = setup(tmp_path / "three")
    ctx3.staging.file_candidate(entry_candidate("x claim", h3, 0, 6), T0)
    (tmp_path / "three" / "data" / "robotics" / "handbook.txt").write_bytes(b"tampered")
    _, t3 = run_session(ctx3, ["q"])
    assert "SOURCE INVALID" in t3 and "does not hash" in t3


def test_f5_unchecked_approval_is_refused_unless_allowed_and_then_recorded(tmp_path):
    def failing(content, layer, pack):
        raise GateError("retriever ERROR: boom")

    ctx, h, text = setup(tmp_path, related=failing)
    ctx.staging.file_candidate(entry_candidate("x claim", h, 0, 6), T0)
    summary, transcript = run_session(ctx, ["a", "q"])
    assert "RELATED LOOKUP FAILED" in transcript and "REFUSED" in transcript
    assert summary.counts["approve"] == 0
    assert summary.counts["candidates_display_unavailable"] == 1
    assert ctx.decisions.read_all() == []
    # no finder at all: the same refusal
    ctx2, h2, _ = setup(tmp_path / "two", related=None)
    ctx2.staging.file_candidate(entry_candidate("x claim", h2, 0, 6), T0)
    summary2, transcript2 = run_session(ctx2, ["a", "q"])
    assert "NO CONFLICT DISPLAY" in transcript2 and summary2.counts["approve"] == 0
    # allowed explicitly: the approval goes through and the record says so
    ctx3, h3, _ = setup(tmp_path / "three", related=failing, allow_unchecked_approval=True)
    ctx3.staging.file_candidate(entry_candidate("x claim", h3, 0, 6), T0)
    summary3, _ = run_session(ctx3, ["a", "y", "x-claim", "", "q"])
    d = ctx3.decisions.read_all()[0]
    assert d["review"]["conflict_display"] == "lookup-failed"
    assert ctx3.runs.read_all()[0]["config"]["allow_unchecked_approval"] is True
    assert ctx3.runs.read_all()[0]["counts"]["decisions_without_display"] == 1
    # rejecting never needs the display
    ctx4, h4, _ = setup(tmp_path / "four", related=None)
    ctx4.staging.file_candidate(entry_candidate("x claim", h4, 0, 6), T0)
    summary4, _ = run_session(ctx4, ["r", "bad", "q"])
    assert summary4.counts["reject"] == 1
    assert ctx4.decisions.read_all()[0]["review"]["conflict_display"] == "no-finder"


def test_f3_crash_repair_also_closes_the_heads_cluster(tmp_path, monkeypatch):
    ctx, h, text = setup(tmp_path)
    st = ctx.staging
    head = st.file_candidate(entry_candidate("Bearing wear claim.", h, 0, 6), T0)
    member = st.file_candidate(entry_candidate("Bearing wear claim (dup).", h, 0, 6), T1)
    run_session(ctx, ["s", "g", head, "q"])  # member grouped under head
    assert st.state_of(member) == "GROUPED"
    real_transition = st.transition

    def crash(*a, **k):
        raise RuntimeError("simulated crash after the decision append")

    monkeypatch.setattr(st, "transition", crash)
    with pytest.raises(RuntimeError, match="simulated crash"):
        run_session(ctx, ["a", "y", "bearing-wear", ""])
    monkeypatch.setattr(st, "transition", real_transition)
    assert len(ctx.decisions.read_all()) == 1 and st.state_of(head) == "PENDING"
    # the aborted session still wrote its manifest, with the reason
    last = ctx.runs.read_all()[-1]
    assert last["aborted"].startswith("RuntimeError: simulated crash")
    summary, transcript = run_session(ctx, ["q"])
    assert "REPAIRED" in transcript and summary.repaired == 2  # head transition + member close
    assert st.state_of(head) == "APPROVED" and st.state_of(member) == "REJECTED-DUPLICATE"
    decs = ctx.decisions.read_all()
    assert len(decs) == 2 and decs[1]["type"] == "reject-duplicate"
    assert decs[1]["head_decision"] == "journal/decisions.jsonl:1"
    assert st.head_decision_of(head) == "journal/decisions.jsonl:1"


def test_f6_refusal_does_not_abort_and_manifest_is_always_written(tmp_path):
    ctx, h, text = setup(tmp_path)
    st = ctx.staging
    st.file_candidate(entry_candidate("Bearing wear claim.", h, 0, 6), T0)
    st.file_candidate(entry_candidate("Another claim.", h, 0, 6), T1)
    run_session(ctx, ["a", "y", "bearing-wear", "", "q"])
    # reuse of a taken id is refused loudly and the candidate is re-shown, then decided
    summary, transcript = run_session(ctx, ["a", "y", "bearing-wear", "",
                                            "a", "y", "another-claim", "", "q"])
    assert "REFUSED" in transcript and "already taken" in transcript
    assert summary.counts["refused"] == 1 and summary.counts["approve"] == 1
    # other refusals: non-current supersession target, invalid id, unconfirmed bound
    ctx2, h2, _ = setup(tmp_path / "two")
    ctx2.staging.file_candidate(entry_candidate("Claim.", h2, 0, 6), T0)
    s2, t2 = run_session(ctx2, ["a", "y", "ghost", "ghost", "a", "y", "Bad Id!", "",
                                "a", "n", "q"])
    assert s2.counts["refused"] == 3 and "quotation bound was not confirmed" in t2
    assert len(ctx2.runs.read_all()) == 1 and ctx2.runs.read_all()[0]["aborted"] is None


def test_f9_retraction_takes_the_final_form_from_the_journal(tmp_path):
    ctx, h, text = setup(tmp_path)
    ctx.staging.file_candidate(entry_candidate("Bearing wear claim.", h, 0, 6), T0)
    run_session(ctx, ["a", "y", "bearing-wear", "", "q"])
    tree = tmp_path / "tree"
    compile_tree(ctx.decisions, tree)
    buf0 = io.StringIO()
    s0 = GateSession(ctx, Console(file=buf0, color_system=None, markup=False), scripted([]))
    with pytest.raises(GateError, match="exactly as recorded"):  # O9
        s0.retract_entry("BEARING-WEAR")
    s0.end()
    # drift the derived tree: the journal, not the file, is the final form's source
    f = tree / "packs/robotics/entries/bearing-wear.json"
    f.write_bytes(f.read_bytes().replace(b"Bearing wear claim.", b"DRIFTED"))
    buf = io.StringIO()
    console = Console(file=buf, record=True, width=100, color_system=None, markup=False)
    s = GateSession(ctx, console, scripted([]))
    ref = s.retract_entry("bearing-wear")
    assert ref.endswith(":2")
    s.end()
    rec = ctx.decisions.read_all()[1]
    assert rec["entry"]["content"] == "Bearing wear claim."
    assert rec["entry"]["status"] == "retracted"
    assert rec["review"]["conflict_display"] == "not-applicable"  # F13
    with pytest.raises(GateError, match="not a current entry"):
        s.retract_entry("bearing-wear")
    compile_tree(ctx.decisions, tree)
    assert not f.exists()
    assert fold(ctx.decisions.read_all(), tmp_path / "f").retired == frozenset({"bearing-wear"})


def test_f7_session_cost_is_linear_in_the_journals(tmp_path):
    """Scaling guard: a few hundred candidates decided in one session; journal
    reads do not grow with the number of decisions, and wall-clock is bounded."""
    ctx, h, text = setup(tmp_path)
    n = 300
    for i in range(n):
        ctx.staging.file_candidate(entry_candidate(f"Claim number {i} about bearings.", h, 0, 6),
                                   T0)
    reads = {"n": 0}
    real = Journal.read_all

    def counting(self):
        reads["n"] += 1
        return real(self)

    import rag4ds.auditlog.journal as jmod
    answers = []
    for i in range(n):
        answers += ["a", "y", f"claim-{i}", ""]
    answers.append("q")
    jmod.Journal.read_all = counting
    try:
        t0 = time.monotonic()
        summary, _ = run_session(ctx, answers)
        elapsed = time.monotonic() - t0
    finally:
        jmod.Journal.read_all = real
    assert summary.counts["approve"] == n
    # the session read the journals a handful of times at start, not per decision;
    # staging.transition + close_cluster re-read the staging journal per write (S3 code),
    # so the bound is linear: well under 4 reads per decision plus a constant
    assert reads["n"] < 4 * n + 20, reads["n"]
    assert elapsed < 60, elapsed  # 300 decisions; the r1 report measured 187 s for 149


def test_states_matches_state_of_and_stranded_member_is_loud(tmp_path):
    ctx, h, text = setup(tmp_path)
    st = ctx.staging
    a = st.file_candidate(entry_candidate("A.", h, 0, 6), T0)
    b = st.file_candidate(entry_candidate("B.", h, 0, 6), T1)
    st.transition(b, "GROUPED", T2, WHO, "x", head=a)
    assert st.states() == {a: ("PENDING", None), b: ("GROUPED", a)}
    assert {k: st.state_of(k) for k in (a, b)} == {a: "PENDING", b: "GROUPED"}
    # forge a decided head with a still-GROUPED member and no decision: the session refuses
    st.transition(a, "REJECTED", T3, WHO, "journal/decisions.jsonl:99")
    with pytest.raises(GateError, match="stranded member"):
        run_session(ctx, ["q"])


def test_queue_filters_context_validation_and_helpers(tmp_path):
    ctx, h, text = setup(tmp_path)
    st = ctx.staging
    st.file_candidate(entry_candidate("R claim.", h, 0, 6), T0)
    st.file_candidate(entry_candidate("G claim.", h, 0, 6, pack="grocery"), T1)
    buf = io.StringIO()
    s = GateSession(ctx, Console(file=buf, color_system=None), scripted([]), pack="grocery")
    assert len(s.queue()) == 1
    with pytest.raises(GateError, match="not a pack code"):
        GateSession(ctx, Console(file=buf), scripted([]), pack="Bad Pack")
    ids = ctx.identities
    with pytest.raises(GateError, match="code_commit"):
        GateContext(ctx.decisions, ctx.staging, ctx.runs, WHO, "zzz", identities=ids)
    with pytest.raises(GateError, match="approver"):
        GateContext(ctx.decisions, ctx.staging, ctx.runs, "  ", COMMIT, identities=ids)
    with pytest.raises(GateError, match="does not name the decision journal"):
        GateContext(ctx.decisions, ctx.staging, ctx.runs, WHO, COMMIT,
                    journal_ref="elsewhere.jsonl", identities=ids)
    with pytest.raises(GateError, match="display_provenance"):  # O11
        GateContext(ctx.decisions, ctx.staging, ctx.runs, WHO, COMMIT, related_finder=clean_finder,
                    identities=ids)
    with pytest.raises(GateError, match="identity registry is required"):  # AR-rag4ds-026 F3
        GateContext(ctx.decisions, ctx.staging, ctx.runs, WHO, COMMIT)
    with pytest.raises(GateError, match="not registered"):
        GateContext(ctx.decisions, ctx.staging, ctx.runs, "someone else", COMMIT, identities=ids)
    assert suggest_id("Bearing wear dominates actuator failure in rotary joints!") == (
        "bearing-wear-dominates-actuator-failure-in")


def test_related_finder_wires_the_retriever_and_provenance(tmp_path):
    class FakeResp:
        def __init__(self, status, entries=(), meta=None):
            self.status, self.entries, self.meta = status, entries, meta or {}

    class FakeIndex:
        manifest = {"index_version": "abc", "embedding_model": {"id": "m", "revision": "r"},
                    "min_relevance": 0.1, "entry_count": 3}

    class FakeRetriever:
        index = FakeIndex()

        def __init__(self, status):
            self.status = status

        def ground(self, req):
            if self.status == "ERROR":
                return FakeResp("ERROR", meta={"error": "boom"})
            from rag4ds.retrieve.contract import GroundedEntry
            e = GroundedEntry("bearing-wear", 1, "concept", "domain", "robotics", "c", 0.5, "s",
                              status="current")
            return FakeResp("OK", entries=(e,))

    found = make_related_finder(FakeRetriever("OK"))("q", "domain", "robotics")
    assert found[0]["id"] == "bearing-wear"
    with pytest.raises(GateError, match="retriever ERROR"):
        make_related_finder(FakeRetriever("ERROR"))("q", "domain", "robotics")
    assert display_provenance_of(FakeRetriever("OK")) == {
        "index_version": "abc", "embedding_model": {"id": "m", "revision": "r"},
        "min_relevance": 0.1, "entry_count": 3}


# --- AR-rag4ds-020 r2: F11, O6, O10 ---


def test_f11_an_unrenderable_span_never_freezes_the_queue(tmp_path):
    """A bad extractor offset first in filing order: the candidate is shown
    with a loud SOURCE INVALID panel, can be rejected (review.source_span
    'invalid'), and the good candidates behind it are decided in the same
    session."""
    ctx, h, text = setup(tmp_path)
    st = ctx.staging
    bad = st.file_candidate(entry_candidate("Bad offsets.", h, 5000, 5100), T0)
    g1 = st.file_candidate(entry_candidate("Good one.", h, 0, 6), T1)
    g2 = st.file_candidate(entry_candidate("Good two.", h, 0, 6), T2)
    answers = [
        "a", "y",                      # approving the bad one is refused (no rendered span)
        "r", "extractor offsets are outside the document",   # so it is rejected
        "a", "y", "good-one", "",
        "a", "y", "good-two", "",
        "q",
    ]
    summary, transcript = run_session(ctx, answers)
    assert "SOURCE INVALID" in transcript and "REFUSED" in transcript
    assert summary.counts["reject"] == 1 and summary.counts["approve"] == 2
    assert summary.counts["refused"] == 1 and summary.counts["unrenderable"] == 0
    assert st.state_of(bad) == "REJECTED"
    assert st.state_of(g1) == "APPROVED" and st.state_of(g2) == "APPROVED"
    rj = [d for d in ctx.decisions.read_all() if d.get("candidate_id") == bad][0]
    assert rj["review"]["source_span"] == "invalid"
    assert summary.counts["candidates_source_unavailable"] == 1
    assert summary.counts["decisions_without_source"] == 1
    # a candidate panel that cannot render at all is parked, counted, and the queue moves on
    st2 = ctx.staging
    broken = st2.file_candidate(entry_candidate("Broken.", h, 0, 6), T3)
    real_show = GateSession.show

    def failing_show(self, cid, rec):
        if cid == broken:
            raise GateError("simulated unrenderable candidate panel")
        return real_show(self, cid, rec)

    GateSession.show = failing_show
    try:
        g3 = st2.file_candidate(entry_candidate("Good three.", h, 0, 6), T4)
        summary2, transcript2 = run_session(ctx, ["a", "y", "good-three", "", "q"])
    finally:
        GateSession.show = real_show
    assert "UNRENDERABLE" in transcript2 and summary2.counts["unrenderable"] == 1
    assert st2.state_of(g3) == "APPROVED" and st2.state_of(broken) == "PENDING"


def test_o6_approval_needs_both_evidences_unless_allowed(tmp_path):
    ctx, h, text = setup(tmp_path, with_source=False)  # display works, span does not
    ctx.staging.file_candidate(entry_candidate("x claim", h, 0, 6), T0)
    summary, transcript = run_session(ctx, ["a", "y", "q"])
    assert summary.counts["approve"] == 0 and "the source span is unavailable" in transcript
    ctx2, h2, _ = setup(tmp_path / "two", with_source=False, allow_unchecked_approval=True)
    ctx2.staging.file_candidate(entry_candidate("x claim", h2, 0, 6), T0)
    summary2, _ = run_session(ctx2, ["a", "y", "x-claim", "", "q"])
    assert summary2.counts["approve"] == 1
    d = ctx2.decisions.read_all()[0]
    assert d["review"]["source_span"] == "unavailable"
    assert ctx2.runs.read_all()[0]["counts"]["decisions_without_source"] == 1


def test_o10_decisions_layer_requires_review_with_candidate_id():
    from rag4ds.gate.decisions import DecisionError, reject
    with pytest.raises(DecisionError, match="must carry a review context"):
        reject("reject", WHO, T0, "text", "src", "why", candidate_id="cand-" + "a" * 16)
    rec = reject("reject", WHO, T0, "text", "src", "why", candidate_id="cand-" + "a" * 16,
                 review={"conflict_display": "no-finder", "related_shown": [], "conflicts": [],
                         "source_span": "invalid", "quotation_bound_confirmed": False})
    assert rec["review"]["source_span"] == "invalid"


# --- AR-rag4ds-020 r3: F14, O12, O14 ---


def test_f14_an_unreadable_document_is_unavailable_not_fatal(tmp_path):
    """A corpus path that exists but cannot be read as a file (here a directory
    - a real OSError on every platform, no monkeypatching of pathlib): the span
    is UNAVAILABLE, the candidate is rejectable, and the next candidate is
    decided in the same session."""
    ctx, h, text = setup(tmp_path)
    st = ctx.staging
    bad = st.file_candidate(entry_candidate("On the unreadable doc.", h, 0, 6), T0)
    good_text = "Another document entirely with its own sentence."
    other = tmp_path / "data" / "robotics" / "other.txt"
    other.write_text(good_text, encoding="utf-8")
    h2 = hashlib.sha256(good_text.encode()).hexdigest()
    Journal(tmp_path / "runs" / "ingest.jsonl").append({
        "kind": "ingest", "hash": h2, "local_path": "robotics/other.txt", "pack": "robotics",
        "schema": 1, "license": "CC-BY", "source": "ptr", "size_bytes": len(good_text),
        "timestamp": "2026-08-23 13:00 JST", "tool": "t", "code_commit": COMMIT, "config": {}})
    good = st.file_candidate(entry_candidate("On the readable doc.", h2, 0, 7), T1)
    real_resolver = ctx.source_resolver
    unreadable = tmp_path / "data" / "robotics"  # a directory: read_bytes raises OSError

    def resolver(source_hash):
        return unreadable if source_hash == h else real_resolver(source_hash)

    ctx.source_resolver = resolver
    summary, transcript = run_session(ctx, ["a", "y", "r", "unreadable source",
                                            "a", "y", "readable-claim", "", "q"])
    # wrap-proof assertions (rich folds long paths at platform-dependent widths)
    assert "SOURCE UNAVAILABLE" in transcript and "cannot" in transcript
    assert summary.aborted is None and summary.counts["reject"] == 1
    assert summary.counts["approve"] == 1 and summary.counts["refused"] == 1
    assert st.state_of(bad) == "REJECTED" and st.state_of(good) == "APPROVED"
    rj = [d for d in ctx.decisions.read_all() if d.get("candidate_id") == bad][0]
    assert rj["review"]["source_span"] == "unavailable"


def test_o12_mechanical_reviews_share_no_mutable_state():
    from rag4ds.gate.cli import not_applicable_review
    a, b = not_applicable_review(), not_applicable_review()
    a["related_shown"].append("leaked")
    assert b["related_shown"] == [] and not_applicable_review()["related_shown"] == []


def test_o14_invalid_span_is_never_approvable_even_when_unchecked_is_allowed(tmp_path):
    ctx, h, text = setup(tmp_path, allow_unchecked_approval=True)
    ctx.staging.file_candidate(entry_candidate("Bad offsets.", h, 5000, 5100), T0)
    summary, transcript = run_session(ctx, ["a", "q"])
    assert summary.counts["approve"] == 0 and summary.counts["refused"] == 1
    assert "REFUSED" in transcript and "INVALID" in transcript
    # tampered document: also invalid, also refused under the flag
    ctx2, h2, _ = setup(tmp_path / "two", allow_unchecked_approval=True)
    ctx2.staging.file_candidate(entry_candidate("x", h2, 0, 6), T0)
    (tmp_path / "two" / "data" / "robotics" / "handbook.txt").write_bytes(b"tampered")
    summary2, transcript2 = run_session(ctx2, ["a", "r", "tampered", "q"])
    assert summary2.counts["approve"] == 0 and summary2.counts["reject"] == 1
    assert "SOURCE INVALID" in transcript2
