"""
Stage 3b: product name and product group builder (showcase edition).

Builds OUT_PRODUCT_NAME and OUT_PRODUCT_GROUP deterministically from the grammar in
``rules/product_name_rules.json``:

    [PREFIX] [CSD?] [PACK_SIZE] [CONTAINER] [VARIANT?]

    NUKA CSD 12 X 355 ML CAN CORE      a cola multipack
    SLURM 473 ML CAN                   an energy single
    BRAWNDO 8 X 591 ML BTTL            a sports-drink multipack

The built name is then ranked against the PRODUCT_NAME / PRODUCT_GROUP dictionaries
with a token-sort similarity, so the model stage receives both the built string and the
closest valid values. The production builder carried a partner's full portfolio logic;
this version keeps the grammar and the interface (``run_with_candidates``) only.
"""

import csv
import difflib
import json
import re
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parent
_RULES = json.load(open(ROOT / "rules" / "product_name_rules.json", encoding="utf-8"))

_CANONICAL_CATEGORY_TO_CODE = {
    "CARBONATED SOFT DRINKS": "CSD",
    "ENERGY": "ENERGY",
    "ISOTONICS": "ISOTONICS",
    "WATER": "WATER",
    "WATER SPARKLING": "WATER",
    "NON CARBONATED BEVERAGES": "NCB",
}
_CONTAINER_WORD = {"GLASS": "BTTL GLASS", "GABLE": "GABLE", "TETRA": "TETRA", "SLEEK": "CAN", "CAN": "CAN", "BTTL": "BTTL"}
_PACK_TO_CONTAINER = {"CAN": "CAN", "BOTTLE - PET": "BTTL", "BOTTLE - GLASS": "GLASS", "TETRA": "TETRA", "GABLE - TOP": "GABLE"}
_COLA_FLAVOURS = {"", "COLA", "ORIGINAL"}


def _u(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return str(v).upper().strip()


def _load_dictionary(attr: str) -> list:
    path = ROOT / "dictionaries" / f"dictionary_{attr}.csv"
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return [rec[f"{attr}_value"] for rec in csv.DictReader(f)]


_NAME_DICT = _load_dictionary("PRODUCT_NAME")
_GROUP_DICT = _load_dictionary("PRODUCT_GROUP")


def _company_code(manufacturer: str) -> Optional[str]:
    best, best_len = None, 0
    for code, needles in _RULES["company_codes"].items():
        if code.startswith("_"):
            continue
        for needle in needles:
            if needle in manufacturer and len(needle) > best_len:
                best, best_len = code, len(needle)
    return best


def _prefix(brand: str, company: Optional[str]) -> Optional[str]:
    for keyword, prefix in _RULES["independent_brand_prefixes"].items():
        if keyword.startswith("_"):
            continue
        if keyword in brand:
            return prefix
    return company or (brand or None)


def _category_code(out_category: str, category_raw: str, merged: str) -> Optional[str]:
    code = _CANONICAL_CATEGORY_TO_CODE.get(out_category)
    if code:
        return code
    for code, needles in _RULES["category_raw_to_code"].items():
        if code.startswith("_"):
            continue
        if any(n in category_raw or n in merged for n in needles):
            return code
    return None


def _container(brand: str, merged: str, size_ml: Optional[int], count: int, pack_content: str) -> str:
    for b, word in _RULES["brand_container_overrides"].items():
        if not b.startswith("_") and b in brand:
            return _CONTAINER_WORD[word]
    if pack_content in _PACK_TO_CONTAINER:
        return _CONTAINER_WORD[_PACK_TO_CONTAINER[pack_content]]
    padded = f" {merged} "
    for word, needles in _RULES["container_keywords"].items():
        if word.startswith("_"):
            continue
        if any(re.search(rf"\b{re.escape(n.strip())}\b", padded) for n in needles):
            return _CONTAINER_WORD[word]
    inf = _RULES["container_size_inference"]
    if size_ml is None:
        return "CAN"
    if size_ml in inf["tall_can_sizes_ml"]:
        return "CAN"
    limit = inf["can_multipack_max_size_ml"] if count > 1 else inf["can_max_size_single_ml"]
    return "CAN" if size_ml <= limit else "BTTL"


def _size_part(size_ml: Optional[int], count: int) -> Optional[str]:
    if size_ml is None:
        return None
    fmt = _RULES["size_format"]
    if count > 1:
        return fmt["multipack_format"].format(count=count, size=size_ml)
    if size_ml >= fmt["large_threshold_ml"]:
        return fmt["single_large_format"].format(size_L=f"{size_ml / 1000:g}")
    return fmt["single_small_format"].format(size=size_ml)


def _variant_suffix(category_code: str, brand: str, flavour: str, size_ml: Optional[int], count: int) -> Optional[str]:
    v = _RULES["variant_suffix"]
    if category_code not in v["apply_to_categories"]:
        return None
    multipack = v["apply_when_count_gt"] < count <= v["apply_when_count_lte"]
    large = size_ml is not None and size_ml >= v["apply_when_size_ml_gte"]
    if not (multipack or large):
        return None
    is_core_brand = any(cb in brand for cb in v["core_brands"])
    return v["core_suffix"] if is_core_brand and flavour in _COLA_FLAVOURS else v["default_suffix"]


def _token_sort_ratio(a: str, b: str) -> float:
    sa, sb = " ".join(sorted(a.split())), " ".join(sorted(b.split()))
    return round(difflib.SequenceMatcher(None, sa, sb).ratio(), 2)


def _candidates(built: Optional[str], dictionary: list, k: int = 5) -> list:
    if not built:
        return []
    scored = sorted(((v, _token_sort_ratio(built, v)) for v in dictionary), key=lambda x: -x[1])
    return [{"value": v, "score": s} for v, s in scored[:k] if s >= 0.5]


def build_names(row: pd.Series) -> dict:
    """Return OUT_PRODUCT_NAME and OUT_PRODUCT_GROUP for one row (None when not buildable)."""
    merged = " ".join(_u(row.get(c)) for c in ("IN_DESCRIPTION_1", "IN_DESCRIPTION_2", "IN_PRODUCT", "IN_SIZE_ML_RAW"))
    brand = _u(row.get("OUT_BRAND"))
    company = _company_code(_u(row.get("IN_MANUFACTURER")))
    prefix = _prefix(brand, company)
    category_code = _category_code(_u(row.get("OUT_CATEGORY")), _u(row.get("IN_CATEGORY_RAW")), merged)
    if not prefix or not category_code:
        return {"OUT_PRODUCT_NAME": None, "OUT_PRODUCT_GROUP": None}

    size_ml = row.get("parsed_size_ml")
    size_ml = int(size_ml) if size_ml is not None and not pd.isna(size_ml) else None
    count = row.get("parsed_count")
    count = int(count) if count is not None and not pd.isna(count) else 1
    size_part = _size_part(size_ml, count)
    if not size_part:
        return {"OUT_PRODUCT_NAME": None, "OUT_PRODUCT_GROUP": None}

    container = _container(brand, merged, size_ml, count, _u(row.get("OUT_PACK_CONTENT")))
    head = f"{prefix} CSD" if category_code == "CSD" else prefix
    group = f"{head} {size_part} {container}"
    suffix = _variant_suffix(category_code, brand, _u(row.get("OUT_FLAVOUR")), size_ml, count)
    name = f"{group} {suffix}" if suffix else group
    return {"OUT_PRODUCT_NAME": name, "OUT_PRODUCT_GROUP": group}


def run_with_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Add pn_built, pg_built, pn_candidates, pg_candidates columns."""
    out = df.copy()
    built = [build_names(row) for _, row in out.iterrows()]
    out["pn_built"] = [b["OUT_PRODUCT_NAME"] for b in built]
    out["pg_built"] = [b["OUT_PRODUCT_GROUP"] for b in built]
    out["pn_candidates"] = [json.dumps(_candidates(b["OUT_PRODUCT_NAME"], _NAME_DICT)) for b in built]
    out["pg_candidates"] = [json.dumps(_candidates(b["OUT_PRODUCT_GROUP"], _GROUP_DICT)) for b in built]
    return out


if __name__ == "__main__":
    import sys

    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs" / "sample_output.csv"
    df = pd.read_csv(path, dtype=str)
    df["parsed_size_ml"] = pd.to_numeric(df.get("OUT_SIZE_ML"), errors="coerce")
    df["parsed_count"] = pd.to_numeric(df.get("OUT_COUNT"), errors="coerce")
    res = run_with_candidates(df)
    print(res[["IN_DESCRIPTION_1", "pn_built", "pg_built"]].to_string(index=False))
