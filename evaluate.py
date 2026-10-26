"""
Evaluate pipeline output against ground truth.

Compares predicted OUT_* columns against reference truth.
- 8 structured attributes: exact-match accuracy
- PRODUCT_NAME / PRODUCT_GROUP: fuzzy similarity accuracy (token_sort_ratio)
- Combined final score: weighted average of both

Usage:
    python evaluate.py output/result_sample_input_slm.csv truth.csv
    python evaluate.py output/result_your_rows_slm.csv truth.csv
"""

import sys
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz
from eval_scoring import norm, attr_is_correct

OUT_COLS = [
    "OUT_BRAND", "OUT_FLAVOUR", "OUT_CATEGORY", "OUT_SUB_BRAND",
    "OUT_CALORIE", "OUT_PACK_CONTENT", "OUT_SIZE_ML", "OUT_COUNT",
    "OUT_PRODUCT_NAME", "OUT_PRODUCT_GROUP",
]

STRUCTURED_COLS = OUT_COLS[:8]
PN_PG_COLS = OUT_COLS[8:]

FUZZY_THRESHOLD = 80  # token_sort_ratio >= 80 counts as fuzzy-correct


def evaluate(predictions_csv, truth_csv, join_on="IN_UPC"):
    pred = pd.read_csv(predictions_csv, dtype=str)
    truth = pd.read_csv(truth_csv, dtype=str)

    print(f"Predictions: {len(pred)} rows from {predictions_csv}")
    print(f"Truth:       {len(truth)} rows from {truth_csv}")

    if join_on in pred.columns and join_on in truth.columns:
        merged = pred.merge(truth, on=join_on, suffixes=("_pred", "_truth"))
        print(f"Matched:     {len(merged)} rows on {join_on}")
    elif "record_index" in truth.columns or "ROW_ID" in truth.columns:
        if len(pred) != len(truth):
            print(f"Warning: row count mismatch ({len(pred)} vs {len(truth)}), aligning by position")
        n = min(len(pred), len(truth))
        merged = pd.DataFrame()
        for col in OUT_COLS:
            if col in pred.columns:
                merged[f"{col}_pred"] = pred[col].iloc[:n].values
            if col in truth.columns:
                merged[f"{col}_truth"] = truth[col].iloc[:n].values
        print(f"Aligned:     {n} rows by position")
    else:
        print("Error: cannot match predictions to truth", file=sys.stderr)
        sys.exit(1)

    print()
    print("=" * 60)
    print("ATTRIBUTE ACCURACY")
    print("=" * 60)

    results = {}
    pnpg_sim_scores = {}

    # --- 8 structured attributes (exact match) ---
    print("  %-25s %s" % ("ATTRIBUTE", "EXACT MATCH"))
    print("  " + "-" * 45)
    for col in STRUCTURED_COLS:
        pred_col = f"{col}_pred" if f"{col}_pred" in merged.columns else col
        truth_col = f"{col}_truth" if f"{col}_truth" in merged.columns else col
        if pred_col not in merged.columns or truth_col not in merged.columns:
            continue

        correct = 0
        total = len(merged)
        for _, row in merged.iterrows():
            t = norm(row.get(truth_col))
            r = norm(row.get(pred_col))
            if attr_is_correct(t, r, col):
                correct += 1
        pct = correct / total * 100 if total > 0 else 0
        results[col] = {"correct": correct, "total": total, "pct": pct}
        print(f"  {col:<25} {correct:>4}/{total:<4}  ({pct:5.1f}%)")

    struct_correct = sum(results[c]["correct"] for c in STRUCTURED_COLS if c in results)
    struct_total = sum(results[c]["total"] for c in STRUCTURED_COLS if c in results)
    struct_pct = struct_correct / struct_total * 100 if struct_total > 0 else 0
    print("  " + "-" * 45)
    print(f"  {'8 structured attrs':<25} {struct_correct:>4}/{struct_total:<4}  ({struct_pct:5.1f}%)")

    # --- PN / PG (similarity accuracy) ---
    print()
    print("  %-25s %s" % ("ATTRIBUTE", "SIMILARITY"))
    print("  " + "-" * 45)
    for col in PN_PG_COLS:
        pred_col = f"{col}_pred" if f"{col}_pred" in merged.columns else col
        truth_col = f"{col}_truth" if f"{col}_truth" in merged.columns else col
        if pred_col not in merged.columns or truth_col not in merged.columns:
            continue

        scores = []
        total = len(merged)
        for _, row in merged.iterrows():
            t = norm(row.get(truth_col))
            r = norm(row.get(pred_col))
            if t == "" and r == "":
                scores.append(100.0)
            elif t == "" or r == "":
                scores.append(0.0)
            else:
                scores.append(fuzz.token_sort_ratio(t, r))

        avg = sum(scores) / len(scores) if scores else 0
        pnpg_sim_scores[col] = {"avg": avg, "total": total, "scores": scores}
        print(f"  {col:<25} {avg:>5.1f}%")

    pnpg_avg = sum(d["avg"] for d in pnpg_sim_scores.values()) / len(pnpg_sim_scores) if pnpg_sim_scores else 0
    print("  " + "-" * 45)
    print(f"  {'PN + PG similarity':<25} {pnpg_avg:>5.1f}%")

    # --- Combined final accuracy ---
    # 8 structured: exact-match % (weight = 8)
    # PN + PG: similarity % (weight = 2)
    combined = (struct_pct * 8 + pnpg_avg * 2) / 10
    print()
    print("=" * 60)
    print(f"  FINAL COMBINED ACCURACY (all 10 attrs):  {combined:.1f}%")
    print(f"    = (8 x {struct_pct:.1f}% exact  +  2 x {pnpg_avg:.1f}% similarity) / 10")
    print("=" * 60)

    return results, pnpg_sim_scores


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python evaluate.py <predictions.csv> <truth.csv> [--join-on COL]")
        sys.exit(1)

    pred_path = sys.argv[1]
    truth_path = sys.argv[2]
    join_col = "IN_UPC"
    if "--join-on" in sys.argv:
        idx = sys.argv.index("--join-on")
        join_col = sys.argv[idx + 1]

    evaluate(pred_path, truth_path, join_on=join_col)
