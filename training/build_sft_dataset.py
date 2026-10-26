"""
Build the supervised fine-tuning dataset for the stage-4 model.

Runs stages 1 to 3b of the pipeline on an input CSV, pairs every row with its labelled
truth, and writes one chat example per row:

    system    the pipeline's system prompt (valid-value lists included)
    user      the stage-4 payload: raw IN_* fields, top-5 candidates per attribute,
              rule prefills, built product name/group and their candidates
    assistant the ten OUT_* fields as strict JSON

Train and inference therefore see exactly the same prompt. The truth file needs the ten
OUT_* columns and either IN_UPC (joined on) or the same row order as the input.

Usage:
    python training/build_sft_dataset.py data/sample/sample_input.csv docs/sample_output.csv training/sft_sample.jsonl
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import decision_engine  # noqa: E402
import process_input  # noqa: E402
import product_name_builder  # noqa: E402
import rule_prefills  # noqa: E402
import similarity_matcher  # noqa: E402

OUT_FIELDS = [
    "OUT_BRAND", "OUT_FLAVOUR", "OUT_CATEGORY", "OUT_SUB_BRAND", "OUT_CALORIE",
    "OUT_PACK_CONTENT", "OUT_SIZE_ML", "OUT_COUNT", "OUT_PRODUCT_NAME", "OUT_PRODUCT_GROUP",
]


def _top1(raw):
    try:
        c = json.loads(raw) if isinstance(raw, str) else []
        return c[0]["value"] if c else ""
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        return ""


def run_stages_1_to_3b(input_csv: Path):
    """Same stage sequence as raw_pos_parsing.run, without the model."""
    ngrams_df, _ = process_input.process(input_csv)
    ngrams_path = ROOT / "output" / f"ngrams_{input_csv.stem}.csv"
    ngrams_path.parent.mkdir(exist_ok=True)
    ngrams_df.to_csv(ngrams_path, index=False)
    sim_df = similarity_matcher.run_batch(ngrams_path)

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
    return prefill_df, builder_df


def _truth_value(v):
    if v is None or (isinstance(v, float) and pd.isna(v)) or str(v).strip() == "":
        return None
    s = str(v).strip()
    return int(float(s)) if s.replace(".", "", 1).isdigit() else s


def build(input_csv: Path, truth_csv: Path, out_jsonl: Path) -> int:
    prefill_df, builder_df = run_stages_1_to_3b(input_csv)
    truth = pd.read_csv(truth_csv, dtype=str)
    if "IN_UPC" in truth.columns and "IN_UPC" in prefill_df.columns:
        truth = truth.set_index("IN_UPC")
        key = lambda i: prefill_df.at[i, "IN_UPC"]  # noqa: E731
    else:
        key = lambda i: i  # noqa: E731
        truth.index = range(len(truth))

    system_prompt = decision_engine._get_system_prompt()
    n = 0
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for i in range(len(prefill_df)):
            row_data = {col: prefill_df.at[i, col] for col in prefill_df.columns}
            builder_res = {
                "pn_built": builder_df.at[i, "pn_built"] if pd.notna(builder_df.at[i, "pn_built"]) else None,
                "pg_built": builder_df.at[i, "pg_built"] if pd.notna(builder_df.at[i, "pg_built"]) else None,
                "pn_candidates": builder_df.at[i, "pn_candidates"],
                "pg_candidates": builder_df.at[i, "pg_candidates"],
            }
            payload = decision_engine.build_row_payload(row_data, builder_result=builder_res)
            t = truth.loc[key(i)]
            target = {k: _truth_value(t.get(k)) for k in OUT_FIELDS}
            example = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                    {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
                ]
            }
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
            n += 1
    return n


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)
    count = build(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
    print(f"{count} examples written to {sys.argv[3]}")
