"""Offline (air-gapped) posture for a hospital install.

A clinic PC has NO internet. Every model must come from the local bundle
(models/ — see src/utils/paths.py) and nothing may try to download at import,
startup or first request: a HuggingFace fetch on an air-gapped box hangs for
minutes and a failed download used to crash the server lifespan.

    SENTINEL_OFFLINE=1   (default when SENTINEL_HOSPITAL_BUILD=1, or when
                          SENTINEL_DEV_INSECURE is unset — i.e. any non-dev run)
    SENTINEL_OFFLINE=0   developer laptop: HF cache + downloads allowed

When offline, HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1 are exported at IMPORT
time (before transformers / huggingface_hub read them) so every
from_pretrained() resolves strictly from disk, and the model registry turns a
missing model into an explicit 'model missing — run download_all_models.py'
unavailable-reason instead of a hang or a crash.

Import this module BEFORE transformers / huggingface_hub anywhere in src/.
"""

from __future__ import annotations
import os
import sys

HOSPITAL_BUILD: bool = os.environ.get("SENTINEL_HOSPITAL_BUILD") == "1"
_DEV_INSECURE: bool = os.environ.get("SENTINEL_DEV_INSECURE") == "1"
_DEFAULT = "1" if (HOSPITAL_BUILD or not _DEV_INSECURE) else "0"
OFFLINE: bool = os.environ.get("SENTINEL_OFFLINE", _DEFAULT) == "1"

DOWNLOAD_HINT = "run scripts/download_all_models.py on a machine with internet and copy models/ to this install"


def _export_offline_env() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    # Best effort if the libraries were imported before us: their module-level
    # constants were already evaluated from the (old) environment.
    hub = sys.modules.get("huggingface_hub.constants")
    if hub is not None:
        try:
            hub.HF_HUB_OFFLINE = True
        except Exception:
            pass
    tf_hub = sys.modules.get("transformers.utils.hub")
    if tf_hub is not None:
        try:
            tf_hub._is_offline_mode = True
        except Exception:
            pass


if OFFLINE:
    _export_offline_env()


def missing_model_reason(key: str, looked_in: list[str] | None = None) -> str:
    """The unavailable-reason a missing model reports in offline mode (surfaced
    verbatim in /health.models[key].reason)."""
    where = f" (looked in: {', '.join(looked_in)})" if looked_in else ""
    return (f"model missing — run scripts/download_all_models.py --only {key} "
            f"on a machine with internet and copy models/ here{where}")
