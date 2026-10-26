"""
LoRA supervised fine-tuning for the stage-4 model.

Fine-tunes unsloth/Llama-3.2-3B-Instruct on the chat examples written by
build_sft_dataset.py and saves a PEFT adapter the pipeline can load directly
(inference.load_model reads models/<name>/adapter).

The adapter shipped with this repo was trained with PEFT + TRL on a few thousand
labelled rows built from historical mappings. The original training notebook was not
kept, so this script is a faithful reconstruction: the LoRA configuration below is read
straight from the shipped adapter_config.json (rank 32, alpha 16, dropout 0.05, all seven
projection matrices), and the training arguments are the ordinary choices for a 3B model
on a 12 GB GPU rather than the exact originals.

Usage:
    pip install -r training/requirements_training.txt
    python training/build_sft_dataset.py your_rows.csv your_truth.csv training/sft_train.jsonl
    python training/train_lora.py training/sft_train.jsonl --out models/pos-slm-llama3b/adapter
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL = "unsloth/Llama-3.2-3B-Instruct"


def main():
    ap = argparse.ArgumentParser(description="LoRA SFT for the POS CleanScan decision model")
    ap.add_argument("train_jsonl")
    ap.add_argument("--eval-jsonl", default=None)
    ap.add_argument("--out", default=str(ROOT / "models" / "pos-slm-llama3b" / "adapter"))
    ap.add_argument("--base", default=BASE_MODEL)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-seq-len", type=int, default=4096)
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    # Same LoRA shape as the shipped adapter.
    shipped = json.load(open(ROOT / "models" / "pos-slm-llama3b" / "adapter" / "adapter_config.json"))
    lora = LoraConfig(
        r=shipped["r"],
        lora_alpha=shipped["lora_alpha"],
        lora_dropout=shipped["lora_dropout"],
        target_modules=shipped["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 4-bit base so the whole run fits in about 10 GB of VRAM, matching inference.
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base, quantization_config=bnb, device_map="auto")
    model = prepare_model_for_kbit_training(model)

    data_files = {"train": args.train_jsonl}
    if args.eval_jsonl:
        data_files["eval"] = args.eval_jsonl
    ds = load_dataset("json", data_files=data_files)

    cfg = SFTConfig(
        output_dir=str(Path(args.out).parent / "checkpoints"),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_seq_len,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if args.eval_jsonl else "no",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        assistant_only_loss=True,   # learn the JSON answer, not the prompt
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=ds["train"],
        eval_dataset=ds.get("eval"),
        processing_class=tokenizer,
        peft_config=lora,
    )
    trainer.train()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    print(f"adapter saved to {out}")


if __name__ == "__main__":
    main()
