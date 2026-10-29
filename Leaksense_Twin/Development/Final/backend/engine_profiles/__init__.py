"""
Engine profile selection. The single entry point used by main.py/ml/train.py.
Set env var LEAKSENSE_ENGINE to switch engines (default: "c18").
"""

import glob
import os
from .base import EngineProfile

_PROFILES_DIR = os.path.dirname(os.path.abspath(__file__))
_cache = {}


def load_profile(name: str) -> EngineProfile:
    if name not in _cache:
        path = os.path.join(_PROFILES_DIR, f"{name}.json")
        profile = EngineProfile.from_json(path)
        if not profile.calibrated:
            print(f"  [WARN] Engine profile '{name}' is UNCALIBRATED — predictions are not validated.")
        _cache[name] = profile
    return _cache[name]


def active_profile() -> EngineProfile:
    return load_profile(os.environ.get("LEAKSENSE_ENGINE", "c18"))


def list_profile_ids() -> list:
    """All engine profile JSON files available, e.g. ['c18', 'c9_3b']."""
    paths = glob.glob(os.path.join(_PROFILES_DIR, "*.json"))
    return sorted(os.path.splitext(os.path.basename(p))[0] for p in paths)
