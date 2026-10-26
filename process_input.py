"""
Stage 1: Input N-gram Processor

Reads a raw input CSV, builds a single merged word n-gram fingerprint per row.

Each IN_ column's tokens are computed independently then unioned to avoid
false cross-field n-grams.

Output columns:
    <all IN_ columns>   -- original input columns, unchanged
    ngram_token          -- JSON list of union of per-column 1+2+3-gram tokens
"""

import json
import sys
from pathlib import Path

import pandas as pd

from ngram_builder import build_ngram_fingerprint

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

EXCLUDE_COLS = {"IN_SALES", "IN_UNITS", " IN_PRICE ", "IN_PRICE"}


def _row_ngram_token(row: pd.Series, in_cols: list) -> str:
    combined: set = set()
    for col in in_cols:
        val = row.get(col)
        if val is None or (isinstance(val, float) and str(val) in ("nan", "inf")):
            continue
        text = str(val).strip()
        if not text or text.lower() in ("nan", "none", "null", "n/a"):
            continue
        tokens = build_ngram_fingerprint(text)
        combined.update(tokens)
    return json.dumps(sorted(combined))


def process(input_csv: Path, max_rows: int = None):
    print(f"Loading input: {input_csv}")
    df = pd.read_csv(input_csv, dtype=str, low_memory=False, nrows=max_rows)
    print(f"  {len(df)} rows loaded. Building per-column n-gram fingerprints...")

    in_cols = [
        c for c in df.columns
        if c.startswith("IN_") and c.strip() not in EXCLUDE_COLS
    ]

    df["ngram_token"] = df.apply(
        lambda row: _row_ngram_token(row, in_cols), axis=1
    )

    all_in_cols = [c for c in df.columns if c.startswith("IN_")]
    out = df[all_in_cols + ["ngram_token"]].copy()
    return out, in_cols


if __name__ == "__main__":
    input_csv = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    max_rows = int(sys.argv[2]) if len(sys.argv) > 2 else None

    if not input_csv or not input_csv.exists():
        print("Usage: python process_input.py <input.csv> [max_rows]", file=sys.stderr)
        sys.exit(1)

    out_df, in_cols = process(input_csv, max_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_top{max_rows}" if max_rows else "_all"
    out_path = OUTPUT_DIR / f"input_ngrams{suffix}.csv"
    out_df.to_csv(out_path, index=False)

    counts = out_df["ngram_token"].apply(lambda x: len(json.loads(x)))
    print(f"\nN-gram columns used  : {in_cols}")
    print(f"\nFingerprint stats:")
    print(f"  rows             : {len(out_df):,}")
    print(f"  avg tokens/row   : {counts.mean():.1f}")
    print(f"  min tokens/row   : {counts.min()}")
    print(f"  max tokens/row   : {counts.max()}")
    print(f"  empty rows (0)   : {(counts == 0).sum()}")
    print(f"\nOutput -> {out_path}")
