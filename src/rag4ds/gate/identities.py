"""Gate identities (AR-rag4ds-026 F3): OD-1 is a rule about PEOPLE, so the
gate binds every recorded identity - a draft's author, the gate's approver -
to a declared entry of a small registry, not to the spelling of a free-text
string. A name that is not registered (under its canonical form or a listed
alias) is refused by name, loudly, at draft load and at session start;
nothing unregistered can be recorded as an author or an approver, so a
shortened or alternate spelling of the designee's name cannot slip past the
OD-1 refusal - it never reaches the gate.

An alias adds no right: a listed alias resolves to its identity and then
meets every refusal the canonical name meets (OD-1, the author rule, the
pack scope); an unlisted spelling is refused outright. Only the PI adds an
identity or an alias (identities.yaml at the module root).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from rag4ds.model.yamlstrict import DuplicateKeyError, strict_yaml_load

IDENTITIES_SCHEMA_VERSION = 1
IDENTITIES_FILE = "identities.yaml"
ROLES = ("od1-designee", "od1-alternate", "author", "validator")
# The roles that may approve at the gate (AR-rag4ds-026 F12): the OD-1 designee
# and alternate for the shared layers, a validator for a pack. "author" is an
# authoring identity only - it never approves.
GATING_ROLES = ("od1-designee", "od1-alternate", "validator")
_ID_RE = re.compile(r"[a-z][a-z0-9-]{0,39}")


class IdentityError(ValueError):
    """An unregistered, ambiguous or malformed identity - refused, never guessed."""


def normalise(name: str) -> str:
    """Casefold, strip punctuation, collapse whitespace: the comparison key
    ('T. A. Siriweera' and 't a siriweera' are one key). Initials are NOT
    expanded and words are never dropped - two keys are equal only when every
    word agrees."""
    if not isinstance(name, str):
        raise IdentityError(f"an identity is a string, not {type(name).__name__}")
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", name.casefold()).split())


@dataclass(frozen=True)
class Identity:
    id: str
    name: str
    aliases: tuple[str, ...]
    roles: tuple[str, ...]
    # S9 (AR-rag4ds-026 O17/O22): the pack(s) a validator may gate; empty for
    # every other role. "*" (every pack) is admitted only by IdentityRegistry.of
    # for test registries - the loader refuses it in identities.yaml.
    packs: tuple[str, ...] = ()

    def keys(self) -> tuple[str, ...]:
        return (normalise(self.name),) + tuple(normalise(a) for a in self.aliases)

    def validates(self, pack: str) -> bool:
        """Holds the validator role for this pack (or for every pack)."""
        return "validator" in self.roles and (pack in self.packs or "*" in self.packs)


class IdentityRegistry:
    def __init__(self, identities: list[Identity]):
        if not identities:
            raise IdentityError("the identity registry is empty")
        by_key: dict[str, Identity] = {}
        ids: set[str] = set()
        for ident in identities:
            if not _ID_RE.fullmatch(ident.id):
                raise IdentityError(f"identity id {ident.id!r} must be lowercase [a-z0-9-], "
                                    f"1-40 chars, starting with a letter")
            if ident.id in ids:
                raise IdentityError(f"identity id {ident.id!r} is declared twice")
            ids.add(ident.id)
            for role in ident.roles:
                if role not in ROLES:
                    raise IdentityError(f"identity {ident.id!r}: role {role!r} not one of {ROLES}")
            if ident.packs and "validator" not in ident.roles:
                raise IdentityError(f"identity {ident.id!r}: packs are listed only for the "
                                    f"validator role")
            if "validator" in ident.roles and not ident.packs:
                raise IdentityError(f"identity {ident.id!r}: a validator names the pack(s) it "
                                    f"validates (packs: [...])")
            for raw, key in zip((ident.name,) + ident.aliases, ident.keys(), strict=True):
                if not key:
                    raise IdentityError(f"identity {ident.id!r}: name/alias {raw!r} is blank "
                                        f"after normalisation")
                if key in by_key:
                    raise IdentityError(
                        f"identity {ident.id!r}: {raw!r} normalises to {key!r}, which already "
                        f"names identity {by_key[key].id!r} - a spelling may name one person")
                by_key[key] = ident
        self._by_key = by_key
        self.identities = tuple(identities)

    @classmethod
    def load(cls, path: Path) -> IdentityRegistry:
        path = Path(path)
        try:
            data = strict_yaml_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError, DuplicateKeyError) as exc:
            raise IdentityError(f"{path}: not a readable identity registry: {exc}") from exc
        if not isinstance(data, dict) or data.get("schema") != IDENTITIES_SCHEMA_VERSION:
            raise IdentityError(f"{path}: schema must be {IDENTITIES_SCHEMA_VERSION}")
        rows = data.get("identities")
        if not isinstance(rows, list):
            raise IdentityError(f"{path}: identities must be a list")
        out = []
        required = {"id", "name", "aliases", "roles"}
        for i, row in enumerate(rows):
            keys = set(row) if isinstance(row, dict) else set()
            if not (required <= keys <= required | {"packs"}):
                raise IdentityError(f"{path}: identities[{i}] must have the keys id, name, "
                                    f"aliases, roles (and optionally packs)")
            if not isinstance(row["id"], str) or not isinstance(row["name"], str):
                raise IdentityError(f"{path}: identities[{i}]: id and name are strings")
            aliases, roles = row["aliases"] or [], row["roles"] or []
            packs = row.get("packs") or []
            if (not isinstance(aliases, list) or not all(isinstance(a, str) for a in aliases)
                    or not isinstance(roles, list) or not all(isinstance(r, str) for r in roles)
                    or not isinstance(packs, list) or not all(isinstance(p, str) for p in packs)):
                raise IdentityError(f"{path}: identities[{i}]: aliases, roles and packs are "
                                    f"lists of strings")
            if "*" in packs:
                raise IdentityError(f"{path}: identities[{i}]: a validator names its packs; "
                                    f"'*' (every pack) is not admitted from the file")
            out.append(Identity(row["id"], row["name"], tuple(aliases), tuple(roles),
                                tuple(packs)))
        return cls(out)

    @classmethod
    def of(cls, *names: str, roles: tuple[str, ...] = ("validator",),
           packs: tuple[str, ...] = ("*",)) -> IdentityRegistry:
        """A registry of canonical names only (tests and tooling); ids derived;
        every name given the same roles (a pack validator by default) and, for
        validators, the same packs ("*" = every pack; never from the file)."""
        p = tuple(packs) if "validator" in roles else ()
        return cls([Identity(f"i{n}", name, (), tuple(roles), p) for n, name in enumerate(names)])

    def resolve(self, name: str) -> Identity:
        key = normalise(name)
        ident = self._by_key.get(key)
        if ident is None:
            known = ", ".join(repr(i.name) for i in self.identities)
            raise IdentityError(
                f"identity {name!r} is not registered (known: {known}); an unregistered "
                f"spelling is refused, not matched - only the PI adds an identity or an alias")
        return ident

    def same_person(self, a: str, b: str) -> bool:
        """Two recorded names denote the same registered identity. Both must
        resolve; an unregistered name raises rather than compares."""
        return self.resolve(a).id == self.resolve(b).id

    def with_role(self, role: str) -> tuple[Identity, ...]:
        if role not in ROLES:
            raise IdentityError(f"role {role!r} not one of {ROLES}")
        return tuple(i for i in self.identities if role in i.roles)
