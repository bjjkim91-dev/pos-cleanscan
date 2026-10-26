"""
Shared eval accuracy rules.

- Non-empty gold: correct iff normalized prediction equals gold.
- Empty gold: correct iff prediction is also empty.
"""

from __future__ import annotations

import pandas as pd


def norm(v):
    if v is None or pd.isna(v) or str(v).strip().upper() in ("", "NAN", "NONE", "NULL"):
        return ""
    s = str(v).strip().upper()
    try:
        f = float(s)
        if f == int(f):
            return str(int(f))
    except (ValueError, TypeError):
        pass
    return s


def attr_is_correct(t_norm: str, r_norm: str, attr: str | None = None) -> bool:
    if t_norm == "":
        return r_norm == ""
    if t_norm == r_norm:
        return True
    if attr == "OUT_FLAVOUR" and t_norm and r_norm:
        bt = set(t_norm.replace("-", " ").split())
        br = set(r_norm.replace("-", " ").split())
        if len(bt) >= 2 and bt == br:
            return True
    return False


def attr_match_int(t_norm: str, r_norm: str, attr: str | None = None) -> int:
    return 1 if attr_is_correct(t_norm, r_norm, attr) else 0
