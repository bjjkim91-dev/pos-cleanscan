"""
SLM Local Inference Engine

Loads the fine-tuned model (LoRA adapter via HuggingFace PEFT) and
provides the call_slm() interface -- takes a row payload dict, returns
a dict with 10 OUT_* attributes.

Includes valid-value snapping from dictionary CSVs.

Usage:
    python inference.py training/sft_sample.jsonl --limit 5
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DICT_DIR = BASE_DIR / "dictionaries"
MODEL_DIR = BASE_DIR / "models"

OUT_ATTRS = [
    "OUT_BRAND", "OUT_FLAVOUR", "OUT_CATEGORY", "OUT_SUB_BRAND",
    "OUT_CALORIE", "OUT_PACK_CONTENT", "OUT_SIZE_ML", "OUT_COUNT",
    "OUT_PRODUCT_NAME", "OUT_PRODUCT_GROUP",
]

DEFAULT_MAX_NEW_TOKENS = 256

SYSTEM_MSG = (
    "You are a POS product normalization engine for the Canadian beverage market. "
    "Given the product data and candidates below, select the correct canonical value "
    "for each of the 10 output attributes. Return strict JSON only. "
    "For OUT_SUB_BRAND, SUB_BRAND_top5[0] may repeat prefill.OUT_SUB_BRAND "
    "(matched_tokens include RULE_PREFILL); prefer it unless the product text clearly "
    "implies a different line."
)

_VALID_VALUES = None
_MODEL = None
_TOKENIZER = None


def _load_valid_values():
    global _VALID_VALUES
    if _VALID_VALUES is not None:
        return _VALID_VALUES

    values = {}
    for attr in OUT_ATTRS:
        dict_attr = attr.replace("OUT_", "")
        csv_path = DICT_DIR / f"dictionary_{dict_attr}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path, dtype=str)
            val_col = df.columns[0]
            values[attr] = set(df[val_col].dropna().str.strip().unique())
        else:
            values[attr] = set()

    _VALID_VALUES = values
    return _VALID_VALUES


def snap_to_valid(attr, value):
    if value is None:
        return None
    vv = _load_valid_values().get(attr, set())
    if not vv:
        return value

    val_str = str(value).strip()
    if val_str.upper() in {v.upper() for v in vv}:
        for v in vv:
            if v.upper() == val_str.upper():
                return v
        return val_str

    from rapidfuzz import process as rfprocess
    match = rfprocess.extractOne(val_str, list(vv), score_cutoff=70)
    if match:
        return match[0]
    return value


def load_model(model_name="pos-slm-llama3b"):
    global _MODEL, _TOKENIZER
    if _MODEL is not None:
        return _MODEL, _TOKENIZER

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel, PeftConfig

    adapter_path = MODEL_DIR / model_name / "adapter"
    if not adapter_path.exists():
        raise FileNotFoundError(f"No adapter found at {adapter_path}.")

    print(f"Loading fine-tuned model from: {adapter_path}")

    peft_cfg = PeftConfig.from_pretrained(str(adapter_path))
    base_model_id = peft_cfg.base_model_name_or_path

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="sdpa",
    )
    _MODEL = PeftModel.from_pretrained(base_model, str(adapter_path))
    _MODEL.eval()
    _TOKENIZER = AutoTokenizer.from_pretrained(str(adapter_path))
    if _TOKENIZER.pad_token is None:
        _TOKENIZER.pad_token = _TOKENIZER.eos_token

    print("  Model loaded and ready for inference")
    return _MODEL, _TOKENIZER


def call_slm(
    payload: dict,
    model=None,
    tokenizer=None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> Dict:
    if model is None or tokenizer is None:
        model, tokenizer = load_model()

    user_msg = json.dumps(payload, ensure_ascii=False)

    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": user_msg},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    import torch
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    content = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return merge_pass1_with_prefill(_parse_and_snap(content), payload)


def _pass1_field_empty(attr: str, val: Any) -> bool:
    if val is None:
        return True
    s = str(val).strip().upper()
    return s in ("", "NULL", "NONE", "NAN")


def merge_pass1_with_prefill(result: Dict, payload: dict) -> Dict:
    """When SLM JSON is truncated or leaves fields null, fall back to deterministic prefill."""
    prefill = payload.get("prefill") or {}
    if not isinstance(prefill, dict) or not prefill:
        return result
    out = dict(result)
    for attr in OUT_ATTRS:
        if not _pass1_field_empty(attr, out.get(attr)):
            continue
        pv = prefill.get(attr)
        if pv is None:
            continue
        if isinstance(pv, str) and pv.strip().upper() in ("", "NULL", "NONE", "NAN"):
            continue
        if attr in ("OUT_SIZE_ML", "OUT_COUNT"):
            try:
                out[attr] = int(float(pv))
            except (ValueError, TypeError):
                continue
        else:
            out[attr] = snap_to_valid(attr, pv)
    return out


def _parse_and_snap(content: str) -> Dict:
    content = content.strip()
    if content.startswith("```"):
        lines = content.split("\n")
        content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        raw = json.loads(content)
    except json.JSONDecodeError:
        import re
        m = re.search(r'\{[^{}]*\}', content, re.DOTALL)
        if m:
            try:
                raw = json.loads(m.group())
            except json.JSONDecodeError:
                return {a: None for a in OUT_ATTRS}
        else:
            return {a: None for a in OUT_ATTRS}

    result = {}
    for attr in OUT_ATTRS:
        val = raw.get(attr)
        if val is None or str(val).strip().upper() in ("", "NULL", "NONE", "NAN"):
            result[attr] = None
        elif attr in ("OUT_SIZE_ML", "OUT_COUNT"):
            try:
                result[attr] = int(float(val))
            except (ValueError, TypeError):
                result[attr] = val
        else:
            result[attr] = snap_to_valid(attr, val)
    return result


def empty_result():
    return {a: None for a in OUT_ATTRS}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl_file", help="JSONL file with test examples")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--model-name", default="pos-slm-llama3b")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
    )
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_name)
    _load_valid_values()

    examples = []
    with open(args.jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
            if len(examples) >= args.limit:
                break

    correct = 0
    total = 0
    for i, ex in enumerate(examples):
        payload = json.loads(ex["messages"][1]["content"])
        truth = json.loads(ex["messages"][2]["content"])
        result = call_slm(
            payload,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
        )

        row_correct = 0
        row_total = 0
        for attr in OUT_ATTRS:
            t = truth.get(attr)
            r = result.get(attr)
            if t is not None:
                row_total += 1
                if str(t).strip().upper() == str(r).strip().upper() if r else False:
                    row_correct += 1
        correct += row_correct
        total += row_total
        print(f"Row {i}: {row_correct}/{row_total} correct")

    print(f"\nOverall: {correct}/{total} ({correct/total*100:.1f}%)" if total else "No data")
