"""
Raw POS Parsing Pipeline

Takes a raw input CSV and produces a standardized output CSV using the
local fine-tuned SLM (Small Language Model).

Stages:
  1   -- N-gram fingerprints
  2   -- Similarity matching (8 attributes)
  3   -- Deterministic rule prefills
  3b  -- Product name/group builder (candidates)
  4   -- Local SLM resolves all 10 OUT_* attributes
  5   -- Post-processing + valid value snapping
  6   -- Evaluation (if truth file available)

Usage:
    python raw_pos_parsing.py input.csv
    python raw_pos_parsing.py input.csv -o results.csv
    python raw_pos_parsing.py input.csv --truth truth.csv
"""

import json
import re
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

import process_input
import similarity_matcher
import rule_prefills
import product_name_builder
import decision_engine
from inference import call_slm, load_model, empty_result, snap_to_valid, DEFAULT_MAX_NEW_TOKENS

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

IN_COLS = [
    "IN_STORE_CHAIN", "IN_UPC", "IN_DESCRIPTION_1", "IN_DESCRIPTION_2",
    "IN_MANUFACTURER", "IN_PRODUCT", "IN_CATEGORY_RAW",
    "IN_SIZE_ML_RAW", "IN_COUNT_RAW",
]

OUT_COLS = [
    "OUT_BRAND", "OUT_FLAVOUR", "OUT_CATEGORY", "OUT_SUB_BRAND",
    "OUT_CALORIE", "OUT_PACK_CONTENT", "OUT_SIZE_ML", "OUT_COUNT",
    "OUT_PRODUCT_NAME", "OUT_PRODUCT_GROUP",
]


def _top1(top5_json, min_score=0.25):
    try:
        cands = json.loads(top5_json) if pd.notna(top5_json) else []
        if cands and cands[0].get("score", 0) >= min_score:
            return cands[0]["value"]
    except (json.JSONDecodeError, TypeError):
        pass
    return ""


def _rules_only_result(row_data, builder_res):
    """Stage 4 without the model: rule prefills first, then the top similarity candidate, then the builder."""
    pf = json.loads(row_data["prefill_json"]) if pd.notna(row_data.get("prefill_json")) else {}
    lr = {}
    for col in OUT_COLS:
        attr = col.replace("OUT_", "")
        val = pf.get(col)
        if val in (None, ""):
            val = _top1(row_data.get(f"{attr}_top5")) or None
        if val in (None, ""):
            if col == "OUT_PRODUCT_NAME":
                val = builder_res.get("pn_built")
            elif col == "OUT_PRODUCT_GROUP":
                val = builder_res.get("pg_built")
            elif col == "OUT_SUB_BRAND":
                val = "CLASSIC"
            elif col == "OUT_COUNT":
                val = 1
        lr[col] = snap_to_valid(col, val) if val not in (None, "") else None
    return lr


def run(input_csv, output_csv=None, model_name="pos-slm-llama3b", max_new_tokens=DEFAULT_MAX_NEW_TOKENS, rules_only=False):
    input_csv = Path(input_csv)
    if not input_csv.exists():
        print(f"Error: {input_csv} not found")
        sys.exit(1)

    stem = input_csv.stem
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if output_csv is None:
        output_csv = OUTPUT_DIR / f"result_{stem}_slm.csv"
    else:
        output_csv = Path(output_csv)

    t0 = time.perf_counter()
    n_input = len(pd.read_csv(input_csv))
    print(f"POS Processing -- SLM Production Run")
    print(f"Input:  {input_csv} ({n_input} rows)")
    print(f"Output: {output_csv}")
    print(f"Model:  {model_name}")
    print()

    print("[1/5] Extracting n-gram fingerprints...")
    ngrams_df, _ = process_input.process(input_csv)
    ngrams_path = OUTPUT_DIR / f"ngrams_{stem}.csv"
    ngrams_df.to_csv(ngrams_path, index=False)
    n = len(ngrams_df)
    print(f"       {n} rows processed")

    print("[2/5] Running similarity matching...")
    sim_df = similarity_matcher.run_batch(ngrams_path)

    print("[3/5] Applying deterministic rules + PN/PG builder...")
    combined = ngrams_df.copy()
    for col in [c for c in sim_df.columns if c.endswith("_top5")]:
        combined[col] = sim_df[col].values
    prefill_df = rule_prefills.run(combined)

    builder_input = prefill_df.copy()
    pf = builder_input["prefill_json"].apply(lambda x: json.loads(x) if pd.notna(x) else {})
    builder_input["parsed_size_ml"] = pf.apply(lambda p: p.get("OUT_SIZE_ML"))
    builder_input["parsed_count"] = pf.apply(lambda p: p.get("OUT_COUNT"))
    brand_pf = pf.apply(lambda p: p.get("OUT_BRAND", ""))
    brand_sim = sim_df.get("BRAND_top5", pd.Series(["[]"] * len(sim_df))).apply(_top1)
    builder_input["OUT_BRAND"] = brand_pf.where(brand_pf != "", brand_sim)
    builder_df = product_name_builder.run_with_candidates(builder_input)
    built = builder_df["pn_built"].notna().sum()
    print(f"       PN/PG candidates for {built}/{n} rows")

    if rules_only:
        print(f"[4/5] Rules-only mode: no model, prefills + top similarity candidate ({n} rows)...", flush=True)
        model = tokenizer = None
    else:
        print(f"[4/5] Local SLM inference ({n} rows)...", flush=True)
        model, tokenizer = load_model(model_name)
    slm_results = []
    t_slm = time.perf_counter()

    for i in range(n):
        row_data = {col: prefill_df.at[i, col] for col in prefill_df.columns}
        builder_res = {
            "pn_built": builder_df.at[i, "pn_built"] if pd.notna(builder_df.at[i, "pn_built"]) else None,
            "pg_built": builder_df.at[i, "pg_built"] if pd.notna(builder_df.at[i, "pg_built"]) else None,
            "pn_candidates": builder_df.at[i, "pn_candidates"],
            "pg_candidates": builder_df.at[i, "pg_candidates"],
        }
        try:
            if rules_only:
                slm_results.append(_rules_only_result(row_data, builder_res))
                continue
            payload = decision_engine.build_row_payload(row_data, builder_result=builder_res)
            lr = call_slm(
                payload,
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
            )
        except Exception as e:
            print(f"       Row {i} error: {e}", flush=True)
            traceback.print_exc()
            lr = empty_result()
        slm_results.append(lr)

        done = i + 1
        if done % 10 == 0 or done == n or done == 1:
            elapsed = time.perf_counter() - t_slm
            rate = done / elapsed if elapsed > 0 else 0
            eta = (n - done) / rate if rate > 0 else 0
            print(f"       {done}/{n} done  ({elapsed:.0f}s elapsed, ETA {eta:.0f}s)", flush=True)

    print("[5/5] Assembling final output...")
    result = pd.DataFrame()
    for col in IN_COLS:
        result[col] = ngrams_df[col] if col in ngrams_df.columns else None

    for col in OUT_COLS:
        result[col] = [lr.get(col) for lr in slm_results]

    _sp_re = re.compile(r"\bSAN\s*PELLEGR|\bSANPELL|\bSPELLEGR", re.IGNORECASE)
    _ez_re = re.compile(r"ESSENZ|ESSNZ", re.IGNORECASE)
    sp_mask = pd.Series(False, index=result.index)
    ez_mask = pd.Series(False, index=result.index)
    for c in ("IN_DESCRIPTION_1", "IN_DESCRIPTION_2"):
        if c in result.columns:
            sp_mask = sp_mask | result[c].fillna("").apply(lambda x: bool(_sp_re.search(str(x))))
            ez_mask = ez_mask | result[c].fillna("").apply(lambda x: bool(_ez_re.search(str(x))))
    sp_final = sp_mask & ~ez_mask
    if sp_final.sum():
        result.loc[sp_final, "OUT_FLAVOUR"] = None
        result.loc[sp_final, "OUT_CALORIE"] = None
        result.loc[sp_final, "OUT_SUB_BRAND"] = None

    for col in OUT_COLS:
        result[col] = result[col].replace({"": None, "nan": None, "NAN": None, "None": None, "null": None})
    for col in ["OUT_SIZE_ML", "OUT_COUNT"]:
        result[col] = pd.to_numeric(result[col], errors="coerce")
        result[col] = result[col].where(result[col].isna(), result[col].astype("Int64"))

    result.to_csv(output_csv, index=False)

    elapsed = time.perf_counter() - t0
    filled = sum(result[c].notna().sum() for c in OUT_COLS)
    total = n * len(OUT_COLS)
    fill_pct = filled / total * 100 if total > 0 else 0

    print()
    print("=" * 50)
    print("COMPLETE")
    print("=" * 50)
    print(f"  Rows processed:  {n}")
    print(f"  Fill rate:       {fill_pct:.1f}%")
    print(f"  Time:            {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Output:          {output_csv}")
    print()

    return result, output_csv


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="POS Processing -- SLM Production Pipeline")
    parser.add_argument("input_csv", help="Path to input CSV")
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("--model-name", default="pos-slm-llama3b")
    parser.add_argument("--truth", default=None, help="Ground truth CSV for evaluation")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=f"Generation cap per row (default {DEFAULT_MAX_NEW_TOKENS})",
    )
    parser.add_argument(
        "--rules-only",
        action="store_true",
        help="Skip the model: resolve every attribute from rule prefills, similarity candidates and the builder (no GPU needed)",
    )
    args = parser.parse_args()

    _, out_path = run(
        args.input_csv,
        output_csv=args.output,
        model_name=args.model_name,
        max_new_tokens=args.max_new_tokens,
        rules_only=args.rules_only,
    )

    truth_path = args.truth
    if truth_path is None:
        data_dir = BASE_DIR / "data"
        stem = Path(args.input_csv).stem
        for candidate in [
            data_dir / f"{stem.replace('_input', '_truth')}.csv",
            data_dir / f"{stem}_truth.csv",
            data_dir / "ground_truth.csv",
        ]:
            if candidate.exists():
                truth_path = candidate
                break

    if truth_path and Path(truth_path).exists():
        print()
        print("=" * 60)
        print("EVALUATION")
        print("=" * 60)
        from evaluate import evaluate
        evaluate(str(out_path), str(truth_path))
    else:
        print("No truth file found -- skipping evaluation.")
        print("  Pass --truth <path> to enable evaluation.")
