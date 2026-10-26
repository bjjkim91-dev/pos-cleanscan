# Training

Two scripts take you from labelled rows to a LoRA adapter the pipeline can load.

| Step | Script | In | Out |
|---|---|---|---|
| 1. Build examples | `build_sft_dataset.py` | input CSV + truth CSV | JSONL of chat examples |
| 2. Fine-tune | `train_lora.py` | the JSONL | `models/pos-slm-llama3b/adapter/` |

## 1. Build the dataset

```bash
python training/build_sft_dataset.py data/sample/sample_input.csv docs/sample_output.csv training/sft_sample.jsonl
```

Stages 1 to 3b run on every input row exactly as they do at inference, so each example's
user turn is the real stage-4 payload: raw fields, top-5 candidates per attribute, rule
prefills, the built product name and group and their candidates. The assistant turn is the
ten `OUT_*` fields from the truth file as strict JSON. The system turn is the pipeline's own
system prompt with the valid-value lists.

`training/sft_sample.jsonl` in this repo is that command's output on the ten fictional rows:
the exact shape the model was trained on, with nothing real in it.

## 2. Train

```bash
pip install -r training/requirements_training.txt
python training/train_lora.py training/sft_train.jsonl --eval-jsonl training/sft_eval.jsonl
```

- Base: `unsloth/Llama-3.2-3B-Instruct`, loaded in 4-bit NF4 so training fits in about 10 GB of VRAM.
- LoRA: rank 32, alpha 16, dropout 0.05 on q, k, v, o, gate, up and down projections. These are read from the shipped `adapter_config.json`, so a retrain has the same shape as the released adapter.
- Loss on the assistant turn only, cosine schedule, 3 epochs by default.
- Output: a PEFT adapter folder. `inference.py` loads it with `PeftModel.from_pretrained`.

## What is reconstruction and what is original

The released adapter was trained with PEFT and TRL on a few thousand labelled rows built
from a beverage company's historical mappings. Those rows cannot be published and the
original training notebook was not kept in the handover package. `train_lora.py` is
therefore a reconstruction: the LoRA configuration is taken from the shipped adapter, the
dataset builder reuses the pipeline's own payload code, and the training arguments are
standard choices rather than the exact originals.
