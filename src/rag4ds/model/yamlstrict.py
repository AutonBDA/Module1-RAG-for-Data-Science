"""YAML loading that refuses a duplicated mapping key (AR-rag4ds-028 O1):
PyYAML's safe_load silently keeps the last of two equal keys, so a manifest
carrying `state: RELEASED` above `state: DRAFT` loaded as DRAFT without a
word. The human-edited files (pack manifests, the identity registry, a
pack's sources file) are read through this loader and a duplicate is a
loud error. Everything else is exactly yaml.safe_load: anchors, aliases and
merge keys (`<<: *base`) work as they do there - the merged pairs are
flattened in first, and an explicit key that overrides a merged one is the
ordinary YAML meaning, not a duplicate (AR-rag4ds-028 F8: the first version
constructed key nodes before flattening and refused every merge key). The
merge key itself is a key like any other: given twice in one mapping it is
refused, where safe_load applies both (AR-rag4ds-057 r8 F11). One
more difference, for the callers' sake (AR-rag4ds-057 r3 F6, r5 F8): a text
PyYAML parses but cannot build is raised as UnbuildableYAMLError, a YAMLError
each caller already refuses, where safe_load lets a bare Python error escape.
The r3 version named the classes a probe had found; one it had not
(OverflowError, a base-60 float of 175 or more parts) escaped, so the
conversion is now the last line of the load, whatever PyYAML raises."""

from __future__ import annotations

import yaml

from rag4ds.model.render import shown


class DuplicateKeyError(ValueError):
    pass


class UnbuildableYAMLError(yaml.YAMLError):
    """AR-rag4ds-057 r3 F6, r5 F8: PyYAML read the text but could not build a
    value from it - an impossible date or time, a tagged number or boolean it
    cannot read, a base-60 float too large for a float, nesting past the
    recursion limit - and said so with a Python error, not a YAMLError. Raised
    as a YAMLError, so every caller already refuses it as its own
    unreadable-file error; the original error is chained."""


_MERGE = "tag:yaml.org,2002:merge"
_VALUE = "tag:yaml.org,2002:value"


class _StrictLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        # AR-rag4ds-057 r3 F6: a node tagged as a mapping that is not one
        # (`!!map q`, `!!set [a, b]`) is PyYAML's own refusal ("expected a
        # mapping node"), made before this pre-pass reads its value as pairs
        if not isinstance(node, yaml.MappingNode):
            return super().construct_mapping(node, deep=deep)
        self._refuse_repeats(node, deep)
        return super().construct_mapping(node, deep=deep)  # flattens merges itself

    def _refuse_repeats(self, node, deep):
        """Duplicates among a mapping's OWN explicit keys, before the merge
        pairs are flattened in (an explicit key overriding a merged one is
        YAML's meaning, not a repeat); and - AR-rag4ds-057 r6 F9 - the same
        check over every mapping it merges, in place, in a merge list,
        anchored, or merged in turn: PyYAML copies a merge source's pairs
        into its parent without building the source, so a source's own repeat
        would otherwise load silently, its last value winning. Two sources
        giving one key stay YAML's meaning: each source is checked on its own.

        AR-rag4ds-057 r8 F11: the merge key itself given twice in one mapping
        is a key given twice. PyYAML would apply both, the later source
        winning where they disagree - where one merge list of the same two
        sources gives the earlier's - so which value a reader sees would turn
        on how the sources were spelt. A merge list is YAML's form for several
        sources, and the refusal says so.

        AR-rag4ds-057 r7 F10: each mapping is checked once, as written.
        PyYAML's flattening rewrites a mapping, and every source flattened
        into it, in place - its merged pairs set beside its own override, the
        merge key gone - so a mapping met again after that (merged a second
        time, or built later through an alias) would show YAML's own
        override as a repeat. A mapping is always checked before anything
        flattens it: the construction that flattens it checked its whole
        merge closure first. A mapping that merges itself is checked once
        and loads as safe_load has it. The checked mappings are kept, not
        only their ids (r8 O21): flattening drops a merged source from its
        parent, and a node freed during the load could hand its id to
        another."""
        checked = self.__dict__.setdefault("_checked_mappings", {})
        if checked.get(id(node)) is node:
            return
        checked[id(node)] = node
        seen = set()
        merged = False
        for key_node, value_node in node.value:
            if key_node.tag == _MERGE:
                if merged:
                    raise DuplicateKeyError(
                        f"duplicate mapping key '<<' at line {key_node.start_mark.line + 1}; "
                        f"several merge sources go in one list, <<: [*a, *b]")
                merged = True
                sources = (value_node.value if isinstance(value_node, yaml.SequenceNode)
                           else [value_node])
                for source in sources:
                    if isinstance(source, yaml.MappingNode):
                        self._refuse_repeats(source, deep)
                continue
            if key_node.tag == _VALUE and isinstance(key_node, yaml.ScalarNode):
                # AR-rag4ds-057 r8 O20: YAML 1.1's value key (a plain `=`);
                # PyYAML's flattening makes it the string it is written as,
                # after this pre-pass, and SafeLoader has no constructor for
                # its tag - so it is read here as that string
                key = key_node.value
            else:
                key = self.construct_object(key_node, deep=deep)
            try:
                hash(key)
            except TypeError:
                # AR-rag4ds-057 r2 F5: a sequence or a mapping as a key is
                # PyYAML's own refusal to make (a ConstructorError, which is a
                # YAMLError), not a TypeError from this pre-pass
                continue
            if key in seen:
                raise DuplicateKeyError(f"duplicate mapping key {shown(key)} at line "
                                        f"{key_node.start_mark.line + 1}")
            seen.add(key)


def strict_yaml_load(text: str):
    """yaml.safe_load that raises DuplicateKeyError on a key given twice in
    any mapping it reads, a merge source included (AR-rag4ds-057 r6 F9), the
    merge key itself included (r8 F11); merge keys and aliases otherwise
    behave as in safe_load, however often a source is merged (r7 F10). A
    text PyYAML parses
    but cannot build raises UnbuildableYAMLError, a YAMLError, never a bare
    Python error (AR-rag4ds-057 r3 F6, r5 F8); PyYAML's own YAMLErrors and
    DuplicateKeyError pass through as they are."""
    try:
        return yaml.load(text, Loader=_StrictLoader)
    except (DuplicateKeyError, yaml.YAMLError):
        raise
    except Exception as exc:  # the last line: whatever else PyYAML raises while building
        raise UnbuildableYAMLError(f"a value that cannot be built "
                                   f"({type(exc).__name__}: {exc})") from exc
