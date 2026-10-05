"""The dev-load marking (AR-rag4ds-059 r1 O1): tools/build_index.py labels a RELEASED
pack "<version>+unreleased" whenever its gated content - entries and mapping artifacts -
is not the content its release manifest hashed, so an entry gated after a release is never
served under the released version's label; a RELEASED pack with no release manifest is
refused. Driven on copies of the live tree's packs/robotics and shared layers, each
released by construction: its release manifest records its own gated-content hash at its
manifest's version, so the tests do not depend on whether the live tree equals its release."""

import importlib.util
import io
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from rag4ds.index.build import IndexBuildError

MODULE_ROOT = Path(__file__).resolve().parents[1]
TS = "2026-10-05 15:30:00 JST"


def _tool():
    spec = importlib.util.spec_from_file_location("build_index_tool_dev",
                                                  MODULE_ROOT / "tools" / "build_index.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FAKE_Embedder:  # noqa: N801 - a clearly named test double (charter section 8)
    model_id, revision = "FAKE_embedder", "FAKE_rev"

    def embed(self, texts):
        out = np.zeros((len(texts), 8), dtype=np.float32)
        for i, t in enumerate(texts):
            out[i, len(t) % 8] = 1.0
        return out


def _copy(tmp_path):
    """A copy of the pack and the shared layers, released as it stands."""
    root = tmp_path / "mod"
    shutil.copytree(MODULE_ROOT / "packs" / "robotics", root / "packs" / "robotics")
    shutil.copytree(MODULE_ROOT / "shared", root / "shared")
    spec = importlib.util.spec_from_file_location("release_pack_dev",
                                                  MODULE_ROOT / "tools" / "release_pack.py")
    rp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rp)
    v = _version(root)
    rel = root / "packs" / "robotics" / "releases" / f"robotics-{v}.json"
    rel.write_text(json.dumps({"version": v, "gated_content_sha256": rp.gated_content_hash(
        root, rp.shipped_files(root, "robotics"))[0]}), encoding="utf-8")
    return root


def _version(root):
    from rag4ds.model.pack import load_pack_manifest
    return load_pack_manifest(root / "packs" / "robotics").version


def _first_entry(root):
    return sorted((root / "packs" / "robotics" / "entries").glob("*.json"))[0]


def test_a_released_tree_is_served_under_its_version(tmp_path):
    root = _copy(tmp_path)
    assert _tool().pack_versions_from_manifests(root) == {"robotics": _version(root)}


def test_a_changed_entry_is_marked_unreleased(tmp_path):
    tool = _tool()
    root = _copy(tmp_path)
    v = _version(root)
    assert tool.pack_versions_from_manifests(root) == {"robotics": v}
    e = _first_entry(root)
    data = json.loads(e.read_text(encoding="utf-8"))
    data["content"] = data["content"] + " (changed after the release)"
    e.write_text(json.dumps(data), encoding="utf-8")
    assert tool.pack_versions_from_manifests(root) == {"robotics": v + "+unreleased"}


def test_an_entry_added_or_removed_is_marked_unreleased(tmp_path):
    tool = _tool()
    root = _copy(tmp_path)
    e = _first_entry(root)
    shutil.copy(e, e.with_name("zz-added-after-release.json"))
    v = _version(root)
    assert tool.pack_versions_from_manifests(root) == {"robotics": v + "+unreleased"}
    root2 = _copy(tmp_path / "b")
    _first_entry(root2).unlink()
    assert tool.pack_versions_from_manifests(root2) == {"robotics": v + "+unreleased"}


def test_a_released_pack_without_its_release_manifest_is_refused(tmp_path):
    root = _copy(tmp_path)
    v = _version(root)
    (root / "packs" / "robotics" / "releases" / f"robotics-{v}.json").unlink()
    with pytest.raises(IndexBuildError, match=f"release manifest .*robotics-{v}.json is missing"):
        _tool().pack_versions_from_manifests(root)


def test_a_pack_not_released_keeps_its_plain_version(tmp_path):
    root = _copy(tmp_path)
    m = root / "packs" / "robotics" / "manifest.yaml"
    m.write_text(m.read_text(encoding="utf-8").replace("state: RELEASED", "state: POPULATING"),
                 encoding="utf-8")
    e = _first_entry(root)
    e.write_text(e.read_text(encoding="utf-8").replace('"content": "', '"content": "X ', 1),
                 encoding="utf-8")
    assert _tool().pack_versions_from_manifests(root) == {"robotics": _version(root)}


def test_the_built_index_carries_the_mark_and_the_build_says_so(tmp_path, monkeypatch):
    tool = _tool()
    monkeypatch.setattr(tool, "head_commit", lambda root: "0000000")  # the copy is not a repo
    root = _copy(tmp_path)
    _first_entry(root).unlink()  # the tree stays canonical (I1); its content is not its release's
    out = io.StringIO()
    assert tool.run(root, tmp_path / "idx", 0.2, embedder=FAKE_Embedder(), stream=out,
                    clock=lambda: TS) == 0, out.getvalue()
    lines = out.getvalue().splitlines()
    v = _version(root)
    assert json.loads(lines[0])["pack_versions"] == {"robotics": v + "+unreleased"}
    assert f"build_index: pack 'robotics' differs from its release {v} - served as " \
           f"{v}+unreleased" in lines
