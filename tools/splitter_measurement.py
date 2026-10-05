"""VT-3b splitter measurement (AR-rag4ds-024 F1; AR-rag4ds-025 F2): the exact
inputs and the exact procedure behind the decision that S7's rule strategy
keeps the blank sentencizer rather than the pinned language model. Run from
the module root:

    python tools/splitter_measurement.py

It splits three recorded samples three ways - the blank sentencizer S7 uses,
the pinned model's dependency parser (its default sentence boundaries), and
the pinned model's own `senter` component - and prints every sentence list
verbatim plus the comparisons the record cites. The third sample
(PROSE_ABBREVIATIONS) is the case the external checkpoint found the record's
general claim silent on: abbreviation-adjacent scientific prose (citations,
figure references, "et al.", unit periods), where the rule sentencizer is
known to fail and the parser is expected to do better. The decision the
record draws is therefore measured on the case where each splitter wins,
not generalised from two samples. The printed output is pasted into
docs/vt3_pins.md, and tests/test_s7b_language_model.py asserts the same
facts from measure(), so a future model that changes them turns the suite
red and forces the decision to be re-taken instead of leaving a stale
justification standing.
"""

from __future__ import annotations

import sys

import spacy

from rag4ds.extract.language_model import load_pinned_language_model

# Byte-identical to SYNTH in tests/test_s7_extraction.py (the S7 corpus; its
# sha256 is the source_hash in docs/s7_precision_check.md).
SYNTH = (
    "Robotics maintenance notes.\n"
    "Bearing wear is the dominant failure mode of rotary actuators.\n"
    "Torque signature: a rising torque trend that precedes bearing failure.\n"
    "Backlash refers to lost motion in a geartrain.\n"
    "The weather was fine and the team went home early.\n"
    "An encoder is a sensor that reports shaft angle.\n"
)

# The heading-and-table sample the VT-3b decision was measured on: an ATX
# heading, an unpunctuated title line, two prose sentences, a markdown table,
# an unpunctuated definition line, and an abbreviation inside a sentence.
HEADING_TABLE = (
    "# Actuator maintenance\n"
    "Overview\n"
    "Bearing wear is the dominant failure mode of rotary actuators. "
    "It is detected early by torque trends.\n"
    "| part | symptom |\n"
    "|---|---|\n"
    "| bearing | rising torque |\n"
    "Backlash refers to lost motion in a geartrain\n"
    "Encoders report shaft angle, e.g. 12.5 deg. They fail rarely.\n"
)

# Abbreviation-adjacent scientific prose (AR-rag4ds-025 F2): a citation with
# "et al." and a year, a figure reference, "et al." ending a sentence, and
# unit periods - eight real sentences, none of which a heading or table.
PROSE_ABBREVIATIONS = (
    "Backlash is lost motion in a geartrain (cf. Paik et al. 2019). "
    "The effect is measurable at low speed.\n"
    "See Fig. 3 for the torque signature. The peak is at 12 Nm.\n"
    "The method follows Smith et al. Results are reproducible.\n"
    "The encoder sampled at 1.5 kHz. Vibration peaked at 3.2 g. "
    "The controller logged every event.\n"
)

SAMPLES = {"SYNTH": SYNTH, "HEADING_TABLE": HEADING_TABLE,
           "PROSE_ABBREVIATIONS": PROSE_ABBREVIATIONS}


def sentencizer_split(text: str) -> list[str]:
    """S7's splitter: the blank English pipeline plus the rule sentencizer."""
    blank = spacy.blank("en")
    blank.add_pipe("sentencizer")
    return [s.text for s in blank(text).sents]


def parser_split(nlp, text: str) -> list[str]:
    """The pinned model as loaded: sentence boundaries set by its parser."""
    return [s.text for s in nlp(text).sents]


_SENTER_NLP = None


def senter_pipeline():
    """A second, separately verified instance of the pinned model with its
    shipped-disabled senter enabled and every boundary-setting or irrelevant
    component off, so senter alone sets the boundaries (pipes: tok2vec,
    senter). select_pipes cannot re-enable a component the model ships
    disabled, hence the explicit enable_pipe."""
    global _SENTER_NLP
    if _SENTER_NLP is None:
        nlp = load_pinned_language_model()
        nlp.enable_pipe("senter")
        for name in ("tagger", "parser", "attribute_ruler", "lemmatizer", "ner"):
            nlp.disable_pipe(name)
        _SENTER_NLP = nlp
    return _SENTER_NLP


def senter_split(text: str) -> list[str]:
    """The pinned model's own senter component as the sole boundary setter."""
    return [s.text for s in senter_pipeline()(text).sents]


def non_empty(sents: list[str]) -> list[str]:
    return [s.strip() for s in sents if s.strip()]


def measure(nlp=None) -> dict:
    """Every split of every sample, plus the derived facts the record cites."""
    nlp = nlp or load_pinned_language_model()
    out: dict = {}
    for label, text in SAMPLES.items():
        s, p, t = sentencizer_split(text), parser_split(nlp, text), senter_split(text)
        out[label] = {
            "sentencizer": s, "parser": p, "senter": t,
            "parser_equals_sentencizer_non_empty": non_empty(p) == non_empty(s),
            "senter_equals_sentencizer_non_empty": non_empty(t) == non_empty(s),
            "parser_splits_inside_a_table_row": any(
                x.strip().startswith("|") and not x.strip().endswith("|") for x in p),
            "counts": {"sentencizer": len(non_empty(s)), "parser": len(non_empty(p)),
                       "senter": len(non_empty(t))},
        }
    return out


def main() -> int:
    nlp = load_pinned_language_model()
    print(f"pinned model: {nlp.meta['lang']}_{nlp.meta['name']} {nlp.meta['version']} "
          f"| spaCy {spacy.__version__} | pipes {nlp.pipe_names}")
    print(f"senter instance pipes: {senter_pipeline().pipe_names}")
    for label, result in measure(nlp).items():
        print(f"\n== {label} ({len(SAMPLES[label])} chars) ==")
        for splitter in ("sentencizer", "parser", "senter"):
            sents = result[splitter]
            print(f"-- {splitter}: {len(sents)} sentences ({len(non_empty(sents))} non-empty)")
            for i, sent in enumerate(sents):
                print(f"   [{i}] {sent!r}")
        print(f"-- parser == sentencizer (non-empty): "
              f"{result['parser_equals_sentencizer_non_empty']}")
        print(f"-- senter == sentencizer (non-empty): "
              f"{result['senter_equals_sentencizer_non_empty']}")
        print(f"-- parser splits inside a table row: "
              f"{result['parser_splits_inside_a_table_row']}")
        print(f"-- non-empty counts: {result['counts']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
