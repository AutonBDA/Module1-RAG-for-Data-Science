"""S9c part 4 - the bound check's source side built once per document
(SourceIndex; AR-rag4ds-030 r1 O1, the PI's decision of 2026-09-14): the
verdicts are the same whether a check is given the text or its index, over
the S7 attack cases and the boundaries; run_extraction builds the index
exactly once per document however many proposals it checks; anything that
is neither text nor an index is refused."""
import sys
from pathlib import Path

import pytest

from rag4ds.extract.strategy import (
    QUOTATION_MAX_CJK_CHARS,
    QUOTATION_MAX_SOURCE_CHARS,
    QUOTATION_MAX_WORDS,
    Document,
    ExtractError,
    Proposal,
    SourceIndex,
    _longest_runs,
    longest_verbatim_run,
    longest_verbatim_runs,
    longest_verbatim_source_chars,
    quotation_bound_check,
    run_extraction,
    tokenise,
    within_quotation_bound,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import test_s7_extraction as s7  # noqa: E402

TS = "2026-09-15 11:00 JST"
COMMIT = "abcdef0"


def _cases():
    """(content, source) pairs: the S7 corpus and scripts, the ceilings at their
    boundaries, interleaved scripts, the unmeasurable, the empty."""
    synth = s7.SYNTH
    at = "a" * QUOTATION_MAX_SOURCE_CHARS
    over = "a" * (QUOTATION_MAX_SOURCE_CHARS + 1)
    words = " ".join(f"w{i}" for i in range(QUOTATION_MAX_WORDS + 3))
    jp = ("\u6a5f\u68b0\u5b66\u7fd2\u306f\u30c7\u30fc\u30bf\u304b\u3089"
          "\u30d1\u30bf\u30fc\u30f3\u3092\u767a\u898b")
    inter = ("immunoprecipitation \u6a5f immunoprecipitation \u68b0 immunoprecipitation \u5b66 "
             "immunoprecipitation \u7fd2")
    pairs = [
        ("bearing wear", synth),
        ("Bearing wear dominates actuator failure in rotary joints.", synth),
        (synth[:200], synth), (synth, synth), ("nothing shared here zz", synth), ("", synth),
        ("bearing", ""), (at, at), (over, over), (words, words), (words, synth),
        (jp, jp), (jp[:QUOTATION_MAX_CJK_CHARS], jp), (inter, inter), (s7.KHMER_RUN, s7.KHMER_RUN),
        (s7.THAI_RUN, s7.THAI_RUN + " " + s7.THAI_RUN), ("!!! ??? ... ***", synth),
        (s7.JP_SOURCE[:30], s7.JP_SOURCE), (s7.RU_SOURCE[:40], s7.RU_SOURCE),
        ("Stall detection - a table row", s7.TABLE_DOC), (s7.TABLE_DOC[:80], s7.TABLE_DOC),
    ]
    return pairs


@pytest.mark.parametrize("content,source", _cases())
def test_the_index_gives_the_same_verdict_and_runs_as_the_text(content, source):
    idx = SourceIndex.build(source)
    assert quotation_bound_check(content, idx) == quotation_bound_check(content, source)
    assert _longest_runs(content, idx) == _longest_runs(content, source)
    assert longest_verbatim_runs(content, idx) == longest_verbatim_runs(content, source)
    assert (longest_verbatim_source_chars(content, idx)
            == longest_verbatim_source_chars(content, source))
    assert longest_verbatim_run(content, idx) == longest_verbatim_run(content, source)
    assert within_quotation_bound(content, idx) == within_quotation_bound(content, source)


def test_the_index_is_bound_to_its_text_and_a_foreign_index_is_refused(monkeypatch):
    """AR-rag4ds-036 F3: the index carries the sha256 of the text it was built
    from; run_extraction refuses an index of another text loudly."""
    import hashlib
    idx = SourceIndex.build(s7.SYNTH)
    assert idx.source_sha256 == hashlib.sha256(s7.SYNTH.encode("utf-8")).hexdigest()
    with pytest.raises(TypeError):
        idx.positions["x"] = (1,)  # read-only (r1 O3)
    doc = Document(source_hash=hashlib.sha256(s7.SYNTH.encode("utf-8")).hexdigest(),
                   text=s7.SYNTH, pack="robotics", layer="domain",
                   source_reference="corpus:robotics/synth")
    foreign = SourceIndex.build(s7.TABLE_DOC)
    monkeypatch.setattr(SourceIndex, "build", staticmethod(lambda text: foreign))
    with pytest.raises(ExtractError, match="does not match the document"):
        run_extraction(doc, [_ManyProposals(3)], COMMIT, TS)


def test_the_type_guard_holds_even_for_unmeasurable_content():
    """AR-rag4ds-036 r1 O1: a non-source is refused before the tokenisable check."""
    with pytest.raises(TypeError, match="source must be text or a SourceIndex"):
        quotation_bound_check("!!! ??? ... ***", 12345)


def test_the_index_is_exactly_what_the_check_computed_per_call_before():
    src = s7.SYNTH
    idx = SourceIndex.build(src)
    toks = tokenise(src)
    assert list(idx.tokens) == toks
    positions = {}
    for j, t in enumerate(toks):
        positions.setdefault(t.text, []).append(j)
    assert {k: list(v) for k, v in idx.positions.items()} == positions
    # the same text builds the same index; a different text does not
    assert SourceIndex.build(src).positions == idx.positions
    assert SourceIndex.build(src + " extra").positions != idx.positions


def test_anything_but_text_or_an_index_is_refused():
    with pytest.raises(TypeError, match="source must be text or a SourceIndex"):
        _longest_runs("x", 3)
    with pytest.raises(TypeError):
        quotation_bound_check("x", ["not", "text"])


class _ManyProposals:
    """A strategy that proposes n copies of the same short content."""
    name = "rule-v1"

    def __init__(self, n):
        self.n = n

    def propose(self, document):
        return [Proposal(type="concept", content=f"Bearing wear {i} - a unit of wear.",
                         span_start=0, span_end=12) for i in range(self.n)]

    def describe(self):
        return {"name": self.name, "params": {"n": self.n}}


def test_run_extraction_builds_the_index_once_per_document(monkeypatch):
    import hashlib
    doc = Document(source_hash=hashlib.sha256(s7.SYNTH.encode("utf-8")).hexdigest(),
                   text=s7.SYNTH, pack="robotics", layer="domain",
                   source_reference="corpus:robotics/synth")
    calls = []
    real_build = SourceIndex.build

    def counting_build(text):
        calls.append(len(text))
        return real_build(text)

    monkeypatch.setattr(SourceIndex, "build", staticmethod(counting_build))
    result = run_extraction(doc, [_ManyProposals(40)], COMMIT, TS)
    assert len(result.candidates) == 40
    assert calls == [len(s7.SYNTH)]  # one build for forty checks
    # the manifest is unchanged in shape and content by the hoist (the id is a
    # hash of the core, which names no index)
    assert "source_index" not in result.manifest
    assert result.manifest["candidates_proposed"] == 40
    assert result.manifest["candidates_suppressed_for_bound"] == 0
