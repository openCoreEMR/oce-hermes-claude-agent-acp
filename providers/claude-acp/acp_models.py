"""The exact model ids this provider offers, and the setting that makes Claude Code offer them.

Under a plan token Claude Code lists only aliases (``sonnet``, ``opus``), and
under a machine's login it lists full ids. The client matches ids exactly, so a
token-mode session needs Claude Code's ``availableModels``. The adapter reads it
only from a settings file and offers those spellings, so ``[1m]`` survives.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

MODELS = (
    "claude-sonnet-5-5",
    "claude-sonnet-5-5[1m]",
    "claude-opus-5-5",
    "claude-opus-5-5[1m]",
    "claude-fable-5-1[1m]",
)
_SETTING = "availableModels"


def _read_settings(path: Path) -> dict:
    try:
        settings = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Cannot read Claude Code's settings at {path}: {exc}") from exc
    if not isinstance(settings, dict):
        raise RuntimeError(f"Claude Code's settings at {path} are not a JSON object.")
    return settings


def pin_available_models(config_dir: str | os.PathLike[str]) -> None:
    """Keep ``MODELS`` in the user settings, leaving every other key alone.

    The rewrite is atomic because concurrent sessions start adapters that read this file.
    """
    path = Path(config_dir) / "settings.json"
    settings = _read_settings(path)
    if settings.get(_SETTING) == list(MODELS):
        return
    content = json.dumps({**settings, _SETTING: list(MODELS)}, indent=2) + "\n"
    temporary = ""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".settings-", suffix=".json")
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
        Path(temporary).replace(path)
    except OSError as exc:
        if temporary:
            Path(temporary).unlink(missing_ok=True)
        raise RuntimeError(f"Cannot write Claude Code's settings at {path}: {exc}") from exc
