"""Suite-wide fixtures. AR-rag4ds-020 O19: when the pinned embedding model is
already in the local Hugging Face cache, run the suite offline so the
real-model tests neither contact the Hub nor pay its latency (the result is
the same either way; only the wall time differed, by ~8x). When it is not
cached the first run must download it, so the variable is left alone."""

import os
from pathlib import Path

PINNED_MODEL_DIR = "models--sentence-transformers--all-MiniLM-L6-v2"
PINNED_REV = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def _snapshot_cached() -> bool:
    # HF_HUB_CACHE overrides the hub directory directly; HF_HOME its parent (O24)
    hub = Path(os.environ.get("HF_HUB_CACHE")
               or Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub")
    snap = hub / PINNED_MODEL_DIR / "snapshots" / PINNED_REV
    return snap.is_dir() and any(snap.iterdir())


if _snapshot_cached() and "HF_HUB_OFFLINE" not in os.environ:
    os.environ["HF_HUB_OFFLINE"] = "1"
