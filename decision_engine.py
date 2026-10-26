"""
Stage 4: Decision Engine -- Payload Builder

Builds the per-row JSON payload that feeds into the SLM.
System prompt includes valid values for all 10 attributes loaded from
dictionary CSVs.

Per-row user prompt includes:
  - IN_* raw fields
  - top5 similarity candidates per attribute
  - prefill deterministic values
  - PN/PG builder candidates (fuzzy-matched from known values)
"""

import json
import re
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DICT_DIR = BASE_DIR / "dictionaries"

LLM_ATTRS = ["BRAND", "FLAVOUR", "CATEGORY", "SUB_BRAND",
             "CALORIE", "PACK_CONTENT", "SIZE_ML", "COUNT",
             "PRODUCT_NAME", "PRODUCT_GROUP"]


def _load_dictionary_values():
    values = {}
    for attr in LLM_ATTRS:
        csv_path = DICT_DIR / f"dictionary_{attr}.csv"
        if not csv_path.exists():
            values[attr] = []
            continue
        df = pd.read_csv(csv_path, dtype=str)
        val_col = df.columns[0]
        values[attr] = sorted(df[val_col].dropna().unique().tolist())
    return values


_DICT_VALUES = None


def _get_dict_values():
    global _DICT_VALUES
    if _DICT_VALUES is None:
        _DICT_VALUES = _load_dictionary_values()
    return _DICT_VALUES


def _build_valid_values_block():
    dv = _get_dict_values()
    lines = []
    for attr in LLM_ATTRS:
        vals = dv.get(attr, [])
        lines.append(f"{attr} ({len(vals)} values):")
        chunk = 20
        for i in range(0, len(vals), chunk):
            lines.append(", ".join(vals[i:i+chunk]))
        lines.append("")
    return "\n".join(lines)


SYSTEM_PROMPT_TEMPLATE = """You are a POS product data normalization engine for the Canadian beverage market.

TASK
For each product row, select the correct canonical value for 10 attributes.
You must ONLY select values from the VALID VALUES lists below.
Do NOT invent, modify, or generate new values.

VALID VALUES
{valid_values}

DECISION PRIORITY

1. PREFILL
   For SIZE_ML, COUNT, CALORIE, PACK_CONTENT, BRAND: prefill is high confidence.
   For FLAVOUR, SUB_BRAND, CATEGORY: treat prefill as a suggestion only.
   Always verify against IN_* text and the rules below.

2. SIMILARITY CANDIDATES
   Each attribute has a top-5 candidate list ranked by score (0-1).
   Score 1.0 = perfect match. Score >= 0.5 = strong. Score < 0.2 = weak.
   Prefer the highest-scoring candidate supported by IN_* text.
   SUB_BRAND only: list[0] may be the rule prefill (matched_tokens include RULE_PREFILL),
   same value as prefill.OUT_SUB_BRAND — prefer it over lower rows unless IN_* text
   clearly indicates a different line.

3. CONTEXTUAL INFERENCE
   If no prefill and no strong candidate, infer from IN_* text.
   The value MUST exist in the VALID VALUES list.
   If truly unknown, output null.

ATTRIBUTE RULES

BRAND: Extract from IN_DESCRIPTION_1/2 and IN_PRODUCT.
  IN_MANUFACTURER is a parent company, NOT the brand.

FLAVOUR:
  The brand name is not a flavour: ignore flavour words that are part of the brand itself.
  Plain or sparkling water: null.

  Compound preference: ALWAYS pick the LONGEST matching compound flavour
  from the VALID VALUES list.

CATEGORY:
  Brand-to-category overrides take priority over IN_CATEGORY_RAW.

SUB_BRAND:
  SUB_BRAND must never be null.
  Default is CLASSIC when no specific sub-brand line is identifiable.

CALORIE:
  ZERO = ZERO SUGAR/DIET/SUGAR FREE/UNSWEETENED/ZS/ZR in text. Plain water = ZERO.
  LOW = LIGHT in text. FULL = everything else.

PACK_CONTENT: CAN/CNS/TIN = CAN, BTTL/BOTTLE/BTL/PET = BOTTLE - PET,
  GLASS/NRG = BOTTLE - GLASS, TETRA = TETRA, GABLE = GABLE - TOP.

SIZE_ML: Extract from NxMML or MML patterns. Convert L to ML.

COUNT: Extract from NxMML pattern or pack indicators. Single = 1. Never null.

PRODUCT_NAME and PRODUCT_GROUP:
  Select from pn_candidates/pg_candidates when available.
  Pick the candidate that best matches the product's BRAND and COMPANY.

OUTPUT FORMAT
Return strict JSON only. No markdown. No explanations.

{{
  "OUT_BRAND": "value or null",
  "OUT_FLAVOUR": "value or null",
  "OUT_CATEGORY": "value or null",
  "OUT_SUB_BRAND": "value (never null, default CLASSIC)",
  "OUT_CALORIE": "value or null",
  "OUT_PACK_CONTENT": "value or null",
  "OUT_SIZE_ML": integer or null,
  "OUT_COUNT": integer (never null, default 1),
  "OUT_PRODUCT_NAME": "value or null",
  "OUT_PRODUCT_GROUP": "value or null"
}}

Every string MUST appear in the VALID VALUES list.
If unknown, output null (except COUNT defaults to 1, SUB_BRAND defaults to CLASSIC)."""

_SYSTEM_PROMPT = None


def _get_system_prompt():
    global _SYSTEM_PROMPT
    if _SYSTEM_PROMPT is None:
        block = _build_valid_values_block()
        _SYSTEM_PROMPT = SYSTEM_PROMPT_TEMPLATE.format(valid_values=block)
    return _SYSTEM_PROMPT


def _to_native(val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if hasattr(val, "item"):
        return val.item()
    return val


def _norm_sb_val(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    s = str(v).strip().upper()
    if s in ("", "NAN", "NONE", "NULL"):
        return ""
    return s


def merge_sub_brand_top5_with_prefill(payload: dict) -> None:
    """
    Put rule-based prefill.OUT_SUB_BRAND first in SUB_BRAND_top5 (score 1.0).
    """
    pf = payload.get("prefill") or {}
    sb = pf.get("OUT_SUB_BRAND")
    sb_u = _norm_sb_val(sb)
    if not sb_u:
        return
    sb_display = str(sb).strip() if isinstance(sb, str) else str(sb).strip()

    raw_list = payload.get("SUB_BRAND_top5")
    if not isinstance(raw_list, list):
        raw_list = []

    filtered = []
    for c in raw_list:
        if not isinstance(c, dict):
            continue
        if _norm_sb_val(c.get("value")) == sb_u:
            continue
        filtered.append(dict(c))

    first = {
        "value": sb_display,
        "score": 1.0,
        "matched_tokens": ["RULE_PREFILL"],
    }
    payload["SUB_BRAND_top5"] = [first] + filtered[:4]


def build_row_payload(row, builder_result=None):
    payload = {}

    skip = {"IN_SALES", "IN_UNITS", "IN_PRICE"}
    for col, val in row.items():
        col_s = str(col)
        if col_s.startswith("IN_") and col_s not in skip:
            v = _to_native(val)
            payload[col_s] = None if v is None else str(v)

    for col, val in row.items():
        col_s = str(col)
        if col_s.endswith("_top5"):
            raw = _to_native(val)
            try:
                payload[col_s] = json.loads(raw) if raw else []
            except (json.JSONDecodeError, TypeError):
                payload[col_s] = []

    prefill_raw = _to_native(row.get("prefill_json", "{}"))
    try:
        payload["prefill"] = json.loads(prefill_raw) if prefill_raw else {}
    except (json.JSONDecodeError, TypeError):
        payload["prefill"] = {}

    if builder_result:
        payload["pn_built"] = builder_result.get("pn_built")
        payload["pg_built"] = builder_result.get("pg_built")
        pn_c = builder_result.get("pn_candidates", [])
        if isinstance(pn_c, str):
            try:
                pn_c = json.loads(pn_c)
            except (json.JSONDecodeError, TypeError):
                pn_c = []
        payload["pn_candidates"] = pn_c[:15]
        pg_c = builder_result.get("pg_candidates", [])
        if isinstance(pg_c, str):
            try:
                pg_c = json.loads(pg_c)
            except (json.JSONDecodeError, TypeError):
                pg_c = []
        payload["pg_candidates"] = pg_c[:10]

    merge_sub_brand_top5_with_prefill(payload)
    return payload


def empty_result():
    return {f"OUT_{a}": None for a in LLM_ATTRS}
