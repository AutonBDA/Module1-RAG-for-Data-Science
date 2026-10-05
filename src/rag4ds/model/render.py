"""Printing a value into a refusal message, never itself a Python error
(AR-rag4ds-057 r4 F7; moved here at r5 so the strict YAML loader, below the
grounding contract, prints through it too)."""

from __future__ import annotations


def shown(value) -> str:
    """repr, except where repr fails - Python will not print an integer of
    more than 4300 decimal digits, nor a value holding one, nor one nested
    past the recursion limit - where the value is named by its type (and an
    integer by its size) instead."""
    try:
        return repr(value)
    except (ValueError, RecursionError):
        if isinstance(value, int):
            return f"<an integer of {value.bit_length()} bits>"
        return f"<a {type(value).__name__} that cannot be printed>"
