"""
Stage 3: rule-based prefills (showcase edition).

Applies deterministic keyword rules to each input row before the model stage and
produces a ``prefill_json`` column: ``{OUT_ATTR: value}`` that the model treats as a
high-confidence default and may only override with strong evidence.

The production rule base held 700+ patterns written against a partner's retailer
data and is not published. This file keeps the same interface and the rule *kinds*,
with a small illustrative set that covers the ten rows in ``data/sample/``:

  OUT_COUNT        -- NxMML pattern extraction; default 1
  OUT_SIZE_ML      -- NxMML / plain numeric extraction, litres converted
  OUT_CALORIE      -- ZERO / DIET / SUGAR FREE keywords; water -> ZERO; else FULL
  OUT_BRAND        -- brand dictionary values and aliases found in the text
  OUT_CATEGORY     -- retailer category text mapped to a canonical category
  OUT_FLAVOUR      -- a handful of flavour keywords
  OUT_SUB_BRAND    -- a handful of sub-brand keywords
  OUT_PACK_CONTENT -- container keywords, then size/count inference
"""

import csv
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent

_NXMML_RE = re.compile(r"(\d+)\s*[Xx]\s*([\d.]+)\s*(ML|L)\b", re.IGNORECASE)
_PLAIN_ML_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(ML|L)\b", re.IGNORECASE)
_PK_RE = re.compile(r"\b(\d+)\s*PK\b", re.IGNORECASE)

_CATEGORY_RULES = [
    ("WATER SPARKLING", r"SPARKLING"),
    ("WATER", r"\bWATER\b"),
    ("ENERGY", r"ENERGY|\bSLURM\b|\bSLRM\b"),
    ("ISOTONICS", r"SPORT|ISOTONIC|ELECTROLYTE|BRAWNDO|BRWNDO"),
    ("NON CARBONATED BEVERAGES", r"ICED? TEA|COFFEE"),
    ("CARBONATED SOFT DRINKS", r"CSD|SOFT DRINK|SODA|\bPOP\b|COLA|NUKA|\bNC\b|SARSAPARILLA|BUZZ"),
]

_FLAVOUR_RULES = [
    ("LEMON LIME", r"LEMON ?LIME|LEM ?LIME"),
    ("SARSAPARILLA", r"SARSAPARILLA|\bSARS\b"),
    ("CHERRY", r"\bCHERRY\b|\bCHRY\b"),
    ("ORANGE", r"\bORANGE\b|\bORNG\b"),
    ("GRAPE", r"\bGRAPE\b|\bGRP\b"),
    ("BERRY", r"\bBERRY\b|\bBRY\b"),
    ("VANILLA", r"\bVANILLA\b|\bVAN\b"),
    ("LEMON", r"\bLEMON\b"),
    ("LIME", r"\bLIME\b"),
    ("COLA", r"\bCOLA\b|\bNUKA\b|\bNC\b"),
    ("ORIGINAL", r"\bORIGINAL\b|\bCLASSIC\b|\bREG\b"),
]

_SUB_BRAND_RULES = [
    ("QUANTUM", r"\bQUANTUM\b|\bQNTM\b"),
    ("THIRST MUTILATOR", r"THIRST MUT|MUTILATOR"),
    ("LOCO", r"\bLOCO\b"),
    ("VICTORY", r"\bVICTORY\b"),
    ("WILD", r"\bWILD\b"),
    ("EXTREME", r"\bX?TREME\b"),
]

_CONTAINER_RULES = [
    ("BOTTLE - GLASS", r"\bGLASS\b|\bNRG\b"),
    ("GABLE - TOP", r"\bGABLE\b|\bCARTON\b"),
    ("TETRA", r"\bTETRA\b"),
    ("CAN", r"\bCANS?\b|\bCNS\b|\bSLEEK\b|\bTIN\b"),
    ("BOTTLE - PET", r"\bBTTL\b|\bBTL\b|\bBOTTLE\b|\bPET\b"),
]


def _norm(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return str(v).upper()


def _merge_text(row: pd.Series) -> str:
    cols = ["IN_DESCRIPTION_1", "IN_DESCRIPTION_2", "IN_PRODUCT", "IN_CATEGORY_RAW", "IN_SIZE_ML_RAW"]
    return " ".join(_norm(row.get(c, "")) for c in cols)


def _load_brand_terms() -> list:
    """(brand value, [search terms]) from the brand dictionary, longest terms first."""
    terms = []
    path = ROOT / "dictionaries" / "dictionary_BRAND.csv"
    if not path.exists():
        return terms
    with open(path, newline="", encoding="utf-8") as f:
        for rec in csv.DictReader(f):
            value = rec["BRAND_value"]
            aliases = json.loads(rec.get("alias_token") or "[]")
            search = {value, value.replace("-", " "), value.replace(" ", "")} | set(aliases)
            terms.append((value, sorted(search, key=len, reverse=True)))
    return terms


_BRAND_TERMS = _load_brand_terms()


def _extract_brand(text: str) -> str:
    compact = re.sub(r"[^A-Z0-9 ]", " ", text)
    best, best_len = "", 0
    for value, search in _BRAND_TERMS:
        for term in search:
            if len(term) < 3:
                continue
            if re.search(rf"\b{re.escape(term)}\b", compact) and len(term) > best_len:
                best, best_len = value, len(term)
    return best


def _rule_count(row: pd.Series):
    merged = _merge_text(row)
    m = _NXMML_RE.search(merged)
    if m:
        return int(m.group(1))
    m = _PK_RE.search(merged)
    if m:
        return int(m.group(1))
    raw = _norm(row.get("IN_COUNT_RAW", ""))
    return int(raw) if raw.isdigit() else 1


def _rule_size_ml(row: pd.Series):
    merged = _merge_text(row)
    m = _NXMML_RE.search(merged) or None
    if m:
        value, unit = float(m.group(2)), m.group(3).upper()
    else:
        m = _PLAIN_ML_RE.search(merged)
        if not m:
            return None
        value, unit = float(m.group(1)), m.group(2).upper()
    return int(round(value * 1000 if unit == "L" else value))


def _rule_calorie(merged: str, category: str) -> str:
    if re.search(r"\bZERO\b|\bDIET\b|SUGAR ?FREE|\bZS\b|\bZR\b|NO SUGAR", merged):
        return "ZERO"
    if category in ("WATER", "WATER SPARKLING"):
        return "ZERO"
    if re.search(r"\bLIGHT\b|\bLOW\b", merged):
        return "LOW"
    return "FULL"


def _first_match(rules, text: str) -> str:
    for value, pattern in rules:
        if re.search(pattern, text):
            return value
    return ""


def _rule_pack_content(merged: str, category: str, size_ml, count) -> str:
    keyword = _first_match(_CONTAINER_RULES, merged)
    if keyword:
        return keyword
    if category == "ENERGY":
        return "CAN"
    if size_ml and size_ml >= 500 and count == 1:
        return "BOTTLE - PET"
    if count and count >= 12 and size_ml and size_ml <= 473:
        return "CAN"
    return ""


def apply_rules(row: pd.Series) -> dict:
    merged = _merge_text(row)
    prefill = {}

    brand = _extract_brand(merged)
    if brand:
        prefill["OUT_BRAND"] = brand

    category = _first_match(_CATEGORY_RULES, _norm(row.get("IN_CATEGORY_RAW", ""))) or _first_match(_CATEGORY_RULES, merged)
    if category:
        prefill["OUT_CATEGORY"] = category

    prefill["OUT_COUNT"] = _rule_count(row)
    size_ml = _rule_size_ml(row)
    if size_ml is not None:
        prefill["OUT_SIZE_ML"] = size_ml

    prefill["OUT_CALORIE"] = _rule_calorie(merged, category)

    flavour = _first_match(_FLAVOUR_RULES, merged)
    if flavour and category not in ("WATER",):
        prefill["OUT_FLAVOUR"] = flavour

    sub_brand = _first_match(_SUB_BRAND_RULES, merged)
    prefill["OUT_SUB_BRAND"] = sub_brand or "CLASSIC"

    pack = _rule_pack_content(merged, category, size_ml, prefill["OUT_COUNT"])
    if pack:
        prefill["OUT_PACK_CONTENT"] = pack

    return prefill


def _inject_prefill_flavour_into_top5(top5_raw, prefill: dict):
    """Put the rule flavour first in FLAVOUR_top5 so similarity noise does not outrank it."""
    flavour = prefill.get("OUT_FLAVOUR")
    if not flavour:
        return top5_raw
    try:
        cands = json.loads(top5_raw) if isinstance(top5_raw, str) else []
    except json.JSONDecodeError:
        cands = []
    cands = [c for c in cands if c.get("value") != flavour]
    cands.insert(0, {"value": flavour, "score": 1.0, "matched_tokens": ["rule"]})
    return json.dumps(cands[:5])


def run(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    prefills = [apply_rules(row) for _, row in out.iterrows()]
    out["prefill_json"] = [json.dumps(p) for p in prefills]
    if "FLAVOUR_top5" in out.columns:
        out["FLAVOUR_top5"] = [
            _inject_prefill_flavour_into_top5(row.get("FLAVOUR_top5"), pf)
            for (_, row), pf in zip(out.iterrows(), prefills)
        ]
    return out


if __name__ == "__main__":
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if not csv_path or not csv_path.exists():
        print("Usage: python rule_prefills.py <input_csv>")
        sys.exit(1)
    df = pd.read_csv(csv_path, dtype=str)
    result = run(df)
    cols = [c for c in ["IN_DESCRIPTION_1", "prefill_json"] if c in result.columns]
    print(result[cols].to_string(index=False))
