"""Workspace locations. Override with the SNAPCALL_HOME environment variable."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def snapcall_home() -> Path:
    p = Path(os.environ.get("SNAPCALL_HOME", Path.home() / ".snapcall")).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p


def runs_dir() -> Path:
    p = snapcall_home() / "runs"
    p.mkdir(exist_ok=True)
    return p


def datasets_dir() -> Path:
    p = snapcall_home() / "datasets"
    p.mkdir(exist_ok=True)
    return p


# --- settings (API keys) -----------------------------------------------------

SECRET_KEYS = ("hf_token", "openai_api_key", "anthropic_api_key")


def _settings_path() -> Path:
    return snapcall_home() / "settings.json"


def load_settings() -> dict[str, Any]:
    p = _settings_path()
    s: dict[str, Any] = json.loads(p.read_text()) if p.exists() else {}
    env = {"hf_token": "HF_TOKEN", "openai_api_key": "OPENAI_API_KEY", "anthropic_api_key": "ANTHROPIC_API_KEY",
           "openai_base_url": "OPENAI_BASE_URL"}
    for k, e in env.items():
        if not s.get(k) and os.environ.get(e):
            s[k] = os.environ[e]
    s.setdefault("openai_base_url", "https://api.openai.com/v1")
    s.setdefault("openai_model", "gpt-4.1-mini")
    s.setdefault("anthropic_model", "claude-opus-5-5")
    s.setdefault("synth_provider", "openai")
    return s


def save_settings(update: dict[str, Any]) -> dict[str, Any]:
    p = _settings_path()
    cur: dict[str, Any] = json.loads(p.read_text()) if p.exists() else {}
    for k, v in update.items():
        if v is None:
            continue
        if k in SECRET_KEYS and isinstance(v, str) and "•" in v:
            continue  # masked value echoed back from the UI, keep the stored one
        cur[k] = v
    p.write_text(json.dumps(cur, indent=2))
    os.chmod(p, 0o600)
    return cur


def masked_settings() -> dict[str, Any]:
    s = load_settings()
    out = {}
    for k, v in s.items():
        if k in SECRET_KEYS and v:
            out[k] = v[:4] + "••••••••" + v[-4:] if len(v) > 10 else "••••••••"
        else:
            out[k] = v
    for k in SECRET_KEYS:
        out.setdefault(k, "")
    return out
