import os
import re
from datetime import datetime
from pathlib import Path

import yaml


def _expand(obj):
    """Expand ${VAR}; list entries whose variable is unset are dropped (optional endpoints)."""
    if isinstance(obj, dict):
        return {k: _expand(v) for k, v in obj.items()}
    if isinstance(obj, list):
        out = [_expand(v) for v in obj]
        return [v for v in out if v is not None]
    if isinstance(obj, str) and "${" in obj:
        missing = [m for m in re.findall(r"\$\{(\w+)\}", obj) if not os.environ.get(m)]
        return None if missing else os.path.expandvars(obj)
    return obj


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        cfg = _expand(yaml.safe_load(f))
    for c in cfg["chains"].values():
        for key in ("start_time", "end_time"):
            if key in c:
                c[key + "_unix"] = int(datetime.fromisoformat(c[key].replace("Z", "+00:00")).timestamp())
    return cfg
