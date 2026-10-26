---
base_model: unsloth/Llama-3.2-3B-Instruct
library_name: peft
pipeline_tag: text-generation
tags:
- lora
- sft
- trl
- llama-3.2
license: llama3.2
---

# pos-slm-llama3b

A LoRA adapter on Llama 3.2 3B Instruct that is the stage-4 decision model of
[POS CleanScan](https://github.com/bjjkim91-dev/pos-cleanscan). It reads one JSON payload per
retailer product line (raw fields, candidate values per attribute, rule prefills, built product
name) and returns the ten canonical attributes as strict JSON.

## Files

| File | Where |
|---|---|
| `adapter_config.json`, `tokenizer_config.json`, `chat_template.jinja`, `adapter_files.sha256` | this folder |
| `adapter_model.safetensors` (195 MB), `tokenizer.json` (17 MB) | [GitHub release `v1.0-adapter`](https://github.com/bjjkim91-dev/pos-cleanscan/releases/tag/v1.0-adapter); fetch with `python models/download_adapter.py` |

## Details

| | |
|---|---|
| Base model | `unsloth/Llama-3.2-3B-Instruct` |
| Method | Supervised fine-tuning with PEFT + TRL |
| LoRA | rank 32, alpha 16, dropout 0.05, on q, k, v, o, gate, up and down projections |
| Inference | 4-bit NF4 through bitsandbytes, greedy decoding, about 10 GB of VRAM |
| Training data | A few thousand labelled rows built from a beverage company's historical product mappings. Not published. |
| Prompt format | Produced by `decision_engine.build_row_payload`; the same code builds training examples (`training/build_sft_dataset.py`) |

## Evaluation

Measured on a held-out set of 100 real rows, exact match on the eight structured attributes:
85.1%. Brand 93%, category 91%, size 89%. Product name and product group are composites and
score 55% exact but 87% token-sort similarity; most misses are formatting, not meaning.

## Limits

The adapter was trained on one company's dictionaries and naming grammar. On other
vocabularies it needs retraining with `training/train_lora.py`. Output quality also depends on
the candidate lists it is given; it does not generate values outside the dictionaries.

## License

Llama 3.2 is used under the Llama 3.2 Community License. The adapter weights follow the same terms.
