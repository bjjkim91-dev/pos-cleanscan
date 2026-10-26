"""
Stage 2: Similarity Matcher

For each input row, computes dictionary-coverage similarity between the row's
ngram_token and every entry in each attribute dictionary.

Returns top-5 matches per OUT_ attribute with:
    value          -- the canonical dictionary value
    score          -- dictionary coverage (0.0-1.0)
    matched_tokens -- tokens in the intersection

PRODUCT_NAME and PRODUCT_GROUP are excluded from this stage.
They are built deterministically from resolved attributes in Stage 3b.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
from rapidfuzz import fuzz as _fuzz

BASE_DIR = Path(__file__).resolve().parent
DICT_DIR = BASE_DIR / "dictionaries"
OUTPUT_DIR = BASE_DIR / "output"
TOP_N = 5

SKIP_ATTRS = {"PRODUCT_NAME", "PRODUCT_GROUP"}


def load_dictionaries() -> Dict[str, pd.DataFrame]:
    dicts = {}
    for f in sorted(DICT_DIR.glob("dictionary_*.csv")):
        attr = f.stem.replace("dictionary_", "")
        if attr in SKIP_ATTRS:
            continue

        df = pd.read_csv(f, dtype=str)
        value_col = df.columns[0]
        df["_ngram"] = df["ngram_token"].apply(
            lambda x: set(json.loads(x)) if pd.notna(x) else set()
        )
        if "alias_token" in df.columns:
            df["_alias"] = df["alias_token"].apply(
                lambda x: set(json.loads(x)) if pd.notna(x) else set()
            )
        else:
            df["_alias"] = [set()] * len(df)

        df["_full"] = df.apply(lambda r: frozenset(r["_ngram"] | r["_alias"]), axis=1)
        df["_value_norm"] = df[value_col].fillna("").astype(str).str.upper() \
                                          .str.replace(r"[^A-Z0-9 ]", " ", regex=True) \
                                          .str.split().str.join(" ")

        dicts[attr] = (value_col, df)
    return dicts


def match_row(
    input_tokens: List[str],
    dictionaries: Dict[str, tuple],
    top_n: int = TOP_N,
) -> Dict[str, List[dict]]:
    """
    Compare input_tokens against every dictionary entry for every attribute.

    Scoring -- dictionary coverage:
        score = |intersection| / |dict_fingerprint|
    """
    input_set = frozenset(input_tokens)
    input_words = [t for t in input_tokens if 3 <= len(t) <= 20 and not t.isdigit()]

    if not input_set:
        return {attr: [] for attr in dictionaries}

    results = {}
    for attr, (value_col, df) in dictionaries.items():
        candidates = []
        for _, row in df.iterrows():
            full_set = row["_full"]
            value_norm = row["_value_norm"]

            ngram_score = 0.0
            matched: set = set()
            if full_set:
                intersection = input_set & full_set
                if intersection:
                    ngram_score = len(intersection) / len(full_set)
                    matched = intersection

            fuzzy_score = 0.0
            if value_norm and input_words:
                dict_words = value_norm.split()
                for iw in input_words:
                    for dw in dict_words:
                        if len(dw) >= 3:
                            r = _fuzz.ratio(iw, dw) / 100.0
                            fuzzy_score = max(fuzzy_score, r)
                val_no_spaces = value_norm.replace(" ", "")
                if len(val_no_spaces) >= 4:
                    for iw in input_words:
                        r = _fuzz.ratio(iw, val_no_spaces) / 100.0
                        fuzzy_score = max(fuzzy_score, r)

            effective_fuzzy = fuzzy_score * 0.85 if fuzzy_score >= 0.75 else 0.0
            score = max(ngram_score, effective_fuzzy)

            if score < 0.25:
                continue

            candidates.append({
                "value": row[value_col],
                "score": round(score, 4),
                "matched_tokens": sorted(matched) if matched else [],
            })

        candidates.sort(key=lambda x: (-x["score"], -len(x["matched_tokens"])))
        results[attr] = candidates[:top_n]

    return results


def run_batch(input_csv: Path, max_rows: int = None) -> pd.DataFrame:
    print(f"Loading input n-grams: {input_csv}")
    df = pd.read_csv(input_csv, dtype=str, nrows=max_rows)
    print(f"  {len(df)} rows. Loading dictionaries...")

    dictionaries = load_dictionaries()
    attrs = sorted(dictionaries.keys())
    print(f"  {len(attrs)} dictionaries loaded: {attrs}")
    print(f"  Running top-{TOP_N} similarity matching...")

    attr_results: Dict[str, List[str]] = {attr: [] for attr in attrs}

    for i, (_, row) in enumerate(df.iterrows()):
        tokens_raw = row.get("ngram_token", "[]")
        input_tokens = json.loads(tokens_raw) if pd.notna(tokens_raw) else []
        matches = match_row(input_tokens, dictionaries)
        for attr in attrs:
            attr_results[attr].append(json.dumps(matches.get(attr, [])))

        if (i + 1) % 50 == 0:
            print(f"    {i + 1}/{len(df)} rows processed...")

    keep_cols = [c for c in ["IN_UPC", "IN_STORE_CHAIN", "IN_DESCRIPTION_1", "IN_DESCRIPTION_2",
                              "IN_MANUFACTURER", "IN_PRODUCT"] if c in df.columns]
    out = df[keep_cols].copy()
    for attr in attrs:
        out[f"{attr}_top5"] = attr_results[attr]

    return out


if __name__ == "__main__":
    input_csv = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    max_rows = int(sys.argv[2]) if len(sys.argv) > 2 else None

    if not input_csv or not input_csv.exists():
        print("Usage: python similarity_matcher.py <ngrams.csv> [max_rows]", file=sys.stderr)
        sys.exit(1)

    out_df = run_batch(input_csv, max_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = input_csv.stem.replace("input_ngrams", "similarity")
    out_path = OUTPUT_DIR / f"{stem}_top5.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nDone. Output -> {out_path}")
