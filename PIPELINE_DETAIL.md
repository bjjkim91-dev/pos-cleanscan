# Pipeline Technical Detail

This document describes what each stage of the Raw POS Parsing pipeline does internally, including the data transformations, algorithms, and decision logic at every step.

---

## Overview

The pipeline takes a CSV of messy retailer POS product data and produces 10 standardized output attributes. It runs 6 stages sequentially, each building on the output of the previous one:

```
raw input CSV
  → [1] N-gram fingerprints
  → [2] Similarity matching against dictionaries
  → [3] Deterministic rule-based prefills
  → [3b] Product name/group construction
  → [4] SLM inference (final attribute resolution)
  → [5] Post-processing and assembly
  → [6] Evaluation (optional)
  → final output CSV
```

Total data flow per row:

| What the row gains at each stage | Format |
|---|---|
| Stage 1: `ngram_token` column | JSON list of string tokens |
| Stage 2: 8 `*_top5` columns | JSON list of `{value, score, matched_tokens}` |
| Stage 3: `prefill_json` column | JSON dict of `{OUT_ATTR: value}` |
| Stage 3b: `pn_built`, `pg_built`, `pn_candidates`, `pg_candidates` | strings + JSON lists |
| Stage 4: 10 `OUT_*` values per row | dict from SLM JSON output |
| Stage 5: cleaned + snapped final `OUT_*` columns | CSV columns |

---

## Stage 1: N-gram Fingerprint Extraction

**Module:** `process_input.py` + `ngram_builder.py`

**Purpose:** Convert every row's free-text fields into a single set of n-gram tokens that can be compared against dictionary entries in Stage 2.

### How it works

1. **Column selection** -- All columns starting with `IN_` are used, except `IN_SALES`, `IN_UNITS`, `IN_PRICE` (financial fields irrelevant to product identity).

2. **Per-column tokenization** -- Each `IN_*` column's value is processed independently:
   - Uppercase the text
   - Strip all non-alphanumeric characters (replace with spaces)
   - Split into word tokens
   - Generate word 1-grams (each word), 2-grams (adjacent pairs concatenated), and 3-grams (adjacent triples concatenated)

   Example for `"SLURM ENERGY 355 ML"`:
   ```
   1-grams: [SLURM, ENERGY, 355, ML]
   2-grams: [SLURMENERGY, ENERGY355, 355ML]
   3-grams: [SLURMENERGY355, ENERGY355ML]
   ```

3. **Union across columns** -- The n-gram sets from all `IN_*` columns are merged (set union). This prevents false cross-field n-grams -- e.g., the last word of `IN_DESCRIPTION_1` won't form a 2-gram with the first word of `IN_DESCRIPTION_2`.

4. **Output** -- Each row gets an `ngram_token` column containing a sorted, deduplicated JSON list of all its n-gram tokens.

### Why n-grams

N-grams capture both exact words and concatenated phrases. Retailer data is notoriously inconsistent -- `"NUKACOLA"`, `"NUKA-COLA"`, and `"NUKA COLA"` all produce overlapping n-grams that can match the canonical dictionary entry. The 2-gram and 3-gram tokens handle cases where spaces are missing or words are glued together in retailer systems.

---

## Stage 2: Similarity Matching

**Module:** `similarity_matcher.py`

**Purpose:** For each row, find the top-5 most similar canonical values from each of 8 attribute dictionaries (BRAND, FLAVOUR, CATEGORY, SUB_BRAND, CALORIE, PACK_CONTENT, SIZE_ML, COUNT).

PRODUCT_NAME and PRODUCT_GROUP are excluded here -- they are built deterministically in Stage 3b.

### How it works

1. **Load dictionaries** -- Each of the 8 `dictionaries/dictionary_*.csv` files contains:
   - The canonical value (first column)
   - A pre-computed `ngram_token` column (the canonical value's own n-gram fingerprint)
   - An optional `alias_token` column (alternative spellings / abbreviations)

   Both token sets are merged into a single `_full` fingerprint set per dictionary entry.

2. **Scoring** -- For each row vs. each dictionary entry, two scores are computed:

   **a) Dictionary-coverage score (primary):**
   ```
   score = |row_tokens ∩ dict_tokens| / |dict_tokens|
   ```
   This measures what fraction of the dictionary entry's fingerprint is present in the row. A score of 1.0 means every token in the dictionary entry was found in the row's text.

   **b) Fuzzy word-match score (fallback):**
   Each word token in the row (3-20 chars, non-numeric) is compared against each word in the dictionary value using `rapidfuzz.fuzz.ratio`. If the best fuzzy score is >= 0.75, it's scaled by 0.85 and used as an alternative score. This catches close misspellings that n-grams miss.

   The final score is `max(ngram_score, fuzzy_score)`. Entries scoring below 0.25 are discarded.

3. **Ranking** -- Candidates are sorted by score (descending), then by number of matched tokens (tiebreaker). The top 5 are kept per attribute.

4. **Output** -- 8 new columns are added: `BRAND_top5`, `FLAVOUR_top5`, `CATEGORY_top5`, `SUB_BRAND_top5`, `CALORIE_top5`, `PACK_CONTENT_top5`, `SIZE_ML_top5`, `COUNT_top5`. Each contains a JSON list like:
   ```json
   [
     {"value": "SLURM", "score": 0.85, "matched_tokens": ["SLURM", "ENERGY", "SLURMENERGY"]},
     {"value": "BRAWNDO", "score": 0.32, "matched_tokens": ["ENERGY"]},
     ...
   ]
   ```

---

## Stage 3: Rule-Based Prefills

**Module:** `rule_prefills.py`

**Purpose:** Apply deterministic keyword rules to resolve as many attributes as possible before the SLM. Rules are high-confidence and fast -- they handle the patterns that regex can solve reliably.

### What it resolves

| Attribute | Rule Logic |
|---|---|
| **OUT_COUNT** | Extract from `NxMML` patterns (e.g. `6X355ML` → 6). Falls back to `IN_COUNT_RAW`, then pack-count hints from text (e.g. `NCB 12PK`). Default: 1. |
| **OUT_SIZE_ML** | Extract from `NxMML` patterns or plain `ML`/`L` values. Converts litres to ml. Parses from `IN_SIZE_ML_RAW`, then `IN_PRODUCT`, then descriptions. |
| **OUT_CALORIE** | Keyword scan: `ZERO SUGAR`, `DIET`, `SUGAR FREE`, `ZS`, `ZR` → `ZERO`. `LIGHT` → `LOW`. Water categories → `ZERO`. Italian MINERAL WATER flavour words → `FULL`. Everything else → `FULL`. |
| **OUT_FLAVOUR** | 370+ regex patterns in priority order, from specific compounds (`MANGO TANGO`, `CHERRY VANILLA`) to single fruits (`ORANGE`, `LIME`). Longest match wins. Brand-specific overrides (e.g., Coke Zero → `CLASSIC`, Sprite plain → `ORIGINAL`). 50+ refinement rules for edge cases. |
| **OUT_SUB_BRAND** | 60+ regex patterns for multi-word sub-brand names (`PURE ZERO`, `ICED COFFEE`, `FRAPPUCCINO`). Brand-context rules (e.g., BIOSTEEL 500ml → `READY TO DRINK`). Heavy refinement pass for product-line corrections. Default: `CLASSIC`. |
| **OUT_PACK_CONTENT** | Keyword match: `CAN`/`CNS`/`TIN` → `CAN`, `BTTL`/`BOTTLE` → `BOTTLE - PET`, `GLASS` → `BOTTLE - GLASS`, etc. Conditional rules when no keyword found: energy → CAN, CSD ≤473ml → CAN, large single-serve → BOTTLE - PET. 30+ refinement rules for premium glass, frozen juice, tetra packs. |
| **OUT_BRAND** | Extracted from descriptions using 100+ keyword/alias pairs (e.g., `BRWNDO` → `BRAWNDO`, `NC` → `NUKA-COLA`). Falls back to top-1 from `BRAND_top5` similarity. |
| **OUT_CATEGORY** | Hierarchical: (1) Brand-override from text (e.g., SLURM → `ENERGY`, BRAWNDO → `ISOTONICS`). (2) Retailer `IN_CATEGORY_RAW` mapping (100+ mappings). (3) Similarity top-5 candidate with context-boosted scoring. Then refinement pass. |

### Special handling

- **Plain mineral water:** FLAVOUR, CALORIE, and SUB_BRAND are cleared (set to null) because these products don't have these attributes.
- **FLAVOUR_top5 injection:** The rule-based flavour is inserted as position 0 in `FLAVOUR_top5` with score 1.0 and tag `RULE_PREFILL`, so the SLM sees it as the strongest candidate.
- **SUB_BRAND_top5 injection:** Same injection pattern for the rule-based sub-brand.

### Output

A new `prefill_json` column is added, containing a JSON dict:
```json
{
  "OUT_BRAND": "SLURM",
  "OUT_FLAVOUR": "ORIGINAL",
  "OUT_CATEGORY": "ENERGY",
  "OUT_SUB_BRAND": "CLASSIC",
  "OUT_CALORIE": "FULL",
  "OUT_PACK_CONTENT": "CAN",
  "OUT_SIZE_ML": 355,
  "OUT_COUNT": 1
}
```

---

## Stage 3b: Product Name & Group Builder

**Module:** `product_name_builder.py`

**Purpose:** Construct `OUT_PRODUCT_NAME` and `OUT_PRODUCT_GROUP` deterministically from the resolved attributes (brand, size, count, container, category), then generate candidate lists by fuzzy-matching against the canonical dictionaries.

### How PRODUCT_NAME is built

The product name follows a structured format:
```
[COMPANY_CODE] [CATEGORY_CODE] [SIZE_STRING] [CONTAINER] [VARIANT_SUFFIX?]
```

**Step 1: Determine company code**

The manufacturer field and brand are mapped to one of:
- `NUKA` (Nuka-Cola Corporation) -- Nuka-Cola and its variants
- `SLURM` (Slurm Corp) -- Slurm
- `BRAWNDO` (Brawndo Corp) -- Brawndo
- `SUNSET`, `BUZZ`, `VIM` -- the other fictional houses in the sample dictionaries
- Independent brands -- use brand name as prefix (e.g., `JARRITOS`, `ZEVIA`)

**Step 2: Determine category code**

Brand-specific overrides from `product_name_rules.json` (e.g., none in the showcase grammar). Falls back to retailer category mapping (e.g., `CARBONATED SOFT DRINKS` → `CSD`).

**Step 3: Format size string**

- Multi-pack: `6 X 355 ML`
- Single ≥1L: `1.5 L`
- Single <1L: `355 ML`

**Step 4: Determine container code**

Uses text keywords, brand overrides from rules, and size-based inference:
- `CAN`, `BTTL` (bottle PET), `GLASS`, `TETRA`, `GABLE`, `POUCH`, `DROPS`

**Step 5: Add variant suffix (optional)**

For CSD products in multi-packs or large sizes, adds `SS` (single-serve) or `FT` (full trade) based on rules.

**Example:** Brand=NUKA-COLA, Category=CSD, Size=6x355ML, Container=CAN → `NUKA CSD 6 X 355 ML CAN`

### How PRODUCT_GROUP is built

Similar to product name but uses a grouped size string and may use different category groupings:
- Energy brands group into `[COMPANY] [size] CAN`, no category word
- NCB brands group into `[COMPANY] NCB [size] [container]`

### Candidate generation

After building the deterministic name, it's fuzzy-matched against all known values in `dictionary_PRODUCT_NAME.csv` and `dictionary_PRODUCT_GROUP.csv` using `token_set_ratio`. The top 10-15 closest matches are returned as candidates with adjusted scores (boosted when the candidate's size/count/container matches the row's resolved values).

### Output

Four new columns:
- `pn_built` -- the deterministic product name (or null if not enough info)
- `pg_built` -- the deterministic product group (or null)
- `pn_candidates` -- JSON list of `{value, score}` top matches from the dictionary
- `pg_candidates` -- JSON list of `{value, score}` top matches from the dictionary

---

## Stage 4: SLM Inference

**Module:** `inference.py` + `decision_engine.py`

**Purpose:** Feed all gathered information (raw fields, similarity candidates, rule prefills, PN/PG candidates) to the fine-tuned language model, which selects the final canonical value for each of the 10 output attributes.

### Payload construction (decision_engine.py)

For each row, `build_row_payload()` assembles a JSON dict:

```json
{
  "IN_STORE_CHAIN": "KWIK-E-MART",
  "IN_UPC": "0000000000010",
  "IN_DESCRIPTION_1": "SLRM ZERO 355 ML",
  "IN_DESCRIPTION_2": "ORG AND NAT - ENERGY DRINKS",
  "IN_MANUFACTURER": "SLURM CORPORATION",
  "IN_CATEGORY_RAW": "ORG AND NAT - ENERGY DRINKS",
  
  "BRAND_top5": [{"value": "SLURM", "score": 0.85, "matched_tokens": [...]}],
  "FLAVOUR_top5": [{"value": "ORIGINAL", "score": 1.0, "matched_tokens": ["RULE_PREFILL"]}],
  "CATEGORY_top5": [...],
  "SUB_BRAND_top5": [...],
  "CALORIE_top5": [...],
  "PACK_CONTENT_top5": [...],
  "SIZE_ML_top5": [...],
  "COUNT_top5": [...],
  
  "prefill": {
    "OUT_BRAND": "SLURM",
    "OUT_FLAVOUR": "ORIGINAL",
    "OUT_CATEGORY": "ENERGY",
    "OUT_CALORIE": "FULL",
    "OUT_PACK_CONTENT": "CAN",
    "OUT_SIZE_ML": 355,
    "OUT_COUNT": 1,
    "OUT_SUB_BRAND": "CLASSIC"
  },
  
  "pn_built": "SLURM 355 ML CAN",
  "pg_built": "SLURM 355 ML CAN",
  "pn_candidates": [{"value": "SLURM 355 ML CAN", "score": 0.95}, ...],
  "pg_candidates": [{"value": "SLURM 355 ML CAN", "score": 0.93}, ...]
}
```

### Model inference (inference.py)

1. **Model loading** -- The LoRA adapter (`models/pos-slm-llama3b/adapter/`) is loaded on top of the base model (`unsloth/Llama-3.2-3B-Instruct`) with 4-bit NF4 quantization. Uses ~10 GB VRAM.

2. **Prompt construction** -- A chat template with:
   - **System message:** Instructs the model to act as a POS normalization engine, select canonical values, and return strict JSON.
   - **User message:** The full payload JSON from above.

3. **Generation** -- `model.generate()` with greedy decoding (`do_sample=False`), max 256 new tokens. Output is deterministic.

4. **Response parsing** -- The model's text output is parsed as JSON. If JSON parsing fails, a regex extracts the first `{...}` block. Each attribute value is then snapped to the closest valid dictionary value using fuzzy matching (score cutoff 70).

5. **Prefill fallback** -- If any attribute is null/empty in the SLM response (e.g., truncated JSON), the deterministic prefill value from Stage 3 is used instead. This ensures SIZE_ML and COUNT are almost never null.

### What the SLM actually decides

The SLM's main job is resolving ambiguities that rules can't handle:
- Choosing between multiple similarity candidates when scores are close
- Picking the right PRODUCT_NAME from 10-15 candidates
- Overriding a prefill when the text clearly implies a different value
- Inferring attributes when no rule pattern matched (rare but possible)

For most rows, the SLM confirms what the rules already determined. Its biggest value is in PRODUCT_NAME and PRODUCT_GROUP selection, where it picks from candidate lists that rules alone can't reliably rank.

---

## Stage 5: Post-Processing & Assembly

**Module:** `raw_pos_parsing.py` (lines 146-174)

**Purpose:** Combine all input columns with SLM results, apply final overrides, clean data types, and write the output CSV.

### Steps

1. **Column assembly** -- Original `IN_*` columns from the input are joined with the 10 `OUT_*` values from the SLM results.

2. **Mineral-water override** -- For any row where `IN_DESCRIPTION_1` or `IN_DESCRIPTION_2` matches a plain mineral-water pattern (flavoured lines excluded), three attributes are forced to null:
   - `OUT_FLAVOUR` → null
   - `OUT_CALORIE` → null
   - `OUT_SUB_BRAND` → null

   This is a domain rule: plain mineral-water products don't have flavours, calorie tiers, or sub-brands.

3. **Null normalization** -- All `OUT_*` columns have empty strings, `"nan"`, `"NAN"`, `"None"`, `"null"` replaced with actual null values.

4. **Numeric casting** -- `OUT_SIZE_ML` and `OUT_COUNT` are converted to nullable integers (`Int64`). Invalid values become null.

5. **Output** -- The final DataFrame is written to `output/result_<stem>_slm.csv`.

6. **Summary stats** -- Fill rate (percentage of non-null cells across all 10 `OUT_*` columns), total time, and output path are printed.

---

## Stage 6: Evaluation

**Module:** `evaluate.py` + `eval_scoring.py`

**Purpose:** Compare pipeline predictions against ground truth and report accuracy metrics.

### Matching rows

Predictions are matched to truth by one of two methods:
1. **UPC join** -- If both files have `IN_UPC`, rows are joined on this column.
2. **Positional alignment** -- If truth has `record_index` or `HOLDOUT_ROW_ID` instead, rows are aligned by position (row 0 to row 0, etc.).

### Normalization

Before comparison, all values are normalized via `eval_scoring.norm()`:
- Strip whitespace, uppercase
- Convert `None`, `NaN`, `null`, empty string → empty
- Numeric values are cast to int when possible (e.g., `"355.0"` → `"355"`)

### Exact-match scoring (8 structured attributes)

For BRAND, FLAVOUR, CATEGORY, SUB_BRAND, CALORIE, PACK_CONTENT, SIZE_ML, COUNT:
- **Non-empty truth:** Correct if normalized prediction equals normalized truth.
- **Empty truth:** Correct only if prediction is also empty.
- **FLAVOUR special case:** Word-set comparison (order-independent) is also accepted for multi-word flavours. E.g., `"CHERRY VANILLA"` matches `"VANILLA CHERRY"`.

### Similarity scoring (PRODUCT_NAME, PRODUCT_GROUP)

For each row, `rapidfuzz.fuzz.token_sort_ratio` computes a 0-100 similarity between the predicted and truth values. This algorithm:
1. Sorts both strings' words alphabetically
2. Computes the edit-distance-based ratio between the sorted versions

This handles word reordering: `"NUKA CSD 355 ML CAN"` vs `"NUKA CSD CAN 355 ML"` score near 100.

Special cases:
- Both empty → 100 (perfect match)
- One empty, other not → 0

The per-attribute average across all rows is reported.

### Combined accuracy

```
Final = (8 × structured_exact_pct + 2 × pn_pg_similarity_avg) / 10
```

This weights the 8 exact-match attributes equally and treats PN + PG as 2 additional attributes, giving a single overall score.

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT CSV                                │
│  IN_STORE_CHAIN, IN_UPC, IN_DESCRIPTION_1, IN_DESCRIPTION_2,   │
│  IN_MANUFACTURER, IN_PRODUCT, IN_CATEGORY_RAW,                 │
│  IN_SIZE_ML_RAW, IN_COUNT_RAW                                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 1: process_input.py + ngram_builder.py                   │
│                                                                 │
│  For each row, for each IN_* column:                            │
│    text → uppercase → strip special chars → word split          │
│    → generate 1-grams, 2-grams, 3-grams                        │
│  Union all columns' tokens → deduplicate → sort                 │
│                                                                 │
│  OUTPUT: ngram_token (JSON list of strings)                     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 2: similarity_matcher.py                                 │
│                                                                 │
│  Load 8 dictionary CSVs (each has pre-computed ngram_token)     │
│  For each row × each dictionary entry:                          │
│    ngram_score = |row ∩ dict| / |dict|                          │
│    fuzzy_score = max word-level fuzz.ratio (if ≥ 0.75)          │
│    score = max(ngram_score, fuzzy_score × 0.85)                 │
│  Keep top 5 per attribute (min score 0.25)                      │
│                                                                 │
│  OUTPUT: BRAND_top5, FLAVOUR_top5, CATEGORY_top5,               │
│          SUB_BRAND_top5, CALORIE_top5, PACK_CONTENT_top5,       │
│          SIZE_ML_top5, COUNT_top5                                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 3: rule_prefills.py                                      │
│                                                                 │
│  200+ keyword regex rules applied per row:                      │
│    COUNT    ← NxMML pattern / IN_COUNT_RAW / pack hints         │
│    SIZE_ML  ← NxMML pattern / IN_SIZE_ML_RAW / description      │
│    CALORIE  ← ZERO/DIET/SUGAR FREE → ZERO, LIGHT → LOW,        │
│               water → ZERO, else FULL                           │
│    FLAVOUR  ← 370+ compound/single-fruit patterns,              │
│               brand-specific overrides, refinement pass          │
│    SUB_BRAND ← 60+ line-name patterns, brand context,           │
│               refinement pass. Default: CLASSIC                  │
│    PACK_CONTENT ← keyword match + conditional rules +            │
│               30+ refinement rules                              │
│    BRAND    ← 100+ keyword/alias patterns from descriptions     │
│    CATEGORY ← brand override → raw mapping → similarity top5    │
│                                                                 │
│  Inject prefill flavour/sub-brand into top5 at position 0       │
│  Plain mineral water: clear FLAVOUR/CALORIE/SUB_BRAND  │
│                                                                 │
│  OUTPUT: prefill_json (JSON dict of resolved OUT_* values)      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 3b: product_name_builder.py                              │
│                                                                 │
│  Build deterministic product name:                              │
│    [COMPANY] [CATEGORY_CODE] [SIZE_STRING] [CONTAINER] [SUFFIX] │
│  Build deterministic product group:                             │
│    Similar but with grouped size/category buckets               │
│                                                                 │
│  Fuzzy-match built names against dictionary entries              │
│  Return top 10-15 candidates ranked by adjusted score           │
│                                                                 │
│  OUTPUT: pn_built, pg_built, pn_candidates, pg_candidates       │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 4: decision_engine.py + inference.py                     │
│                                                                 │
│  Per row:                                                       │
│    1. Build payload: IN_* fields + top5 candidates + prefill    │
│       + pn/pg built + pn/pg candidates                          │
│    2. Send to SLM as chat (system prompt + user JSON)           │
│    3. Model generates JSON with 10 OUT_* values                 │
│    4. Parse response → snap each value to valid dictionary      │
│    5. If any field null → fall back to prefill value             │
│                                                                 │
│  Model: Llama 3.2 3B + LoRA adapter, 4-bit quantized            │
│  Decoding: greedy (deterministic)                               │
│                                                                 │
│  OUTPUT: dict with all 10 OUT_* values per row                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 5: raw_pos_parsing.py (post-processing)                  │
│                                                                 │
│  1. Join IN_* columns with OUT_* results                        │
│  2. Mineral-water override: null FLAVOUR/CALORIE/SUB_BRAND     │
│  3. Normalize nulls: "", "nan", "None", "null" → null           │
│  4. Cast SIZE_ML, COUNT to nullable integers                    │
│  5. Write final CSV                                             │
│                                                                 │
│  OUTPUT: output/result_<name>_slm.csv                           │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 6: evaluate.py + eval_scoring.py                         │
│                                                                 │
│  Match predictions to truth (by UPC or position)                │
│  8 structured attrs: exact match after normalization            │
│  PN + PG: token_sort_ratio similarity (0-100%)                  │
│  Combined: (8 × exact% + 2 × similarity%) / 10                 │
│                                                                 │
│  OUTPUT: printed accuracy report                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Module Reference

| File | Stage | Role |
|------|-------|------|
| `raw_pos_parsing.py` | Orchestrator | Main entry point. Calls all stages, handles CLI args, does post-processing |
| `ngram_builder.py` | 1 | Pure function: text → sorted list of 1+2+3-gram word tokens |
| `process_input.py` | 1 | Reads CSV, applies `ngram_builder` per column per row, unions tokens |
| `similarity_matcher.py` | 2 | Loads 8 dictionary CSVs, computes coverage scores, returns top-5 candidates |
| `rule_prefills.py` | 3 | 200+ regex rules for COUNT, SIZE_ML, CALORIE, FLAVOUR, SUB_BRAND, PACK_CONTENT, BRAND, CATEGORY |
| `product_name_builder.py` | 3b | Constructs canonical PRODUCT_NAME/GROUP from resolved attributes, generates candidate lists |
| `decision_engine.py` | 4 | Assembles the per-row JSON payload sent to the SLM |
| `inference.py` | 4 | Loads model, runs generation, parses JSON, snaps values to dictionaries, prefill fallback |
| `evaluate.py` | 6 | Compares output to truth: exact match (8 attrs) + fuzzy similarity (PN/PG) |
| `eval_scoring.py` | 6 | Shared normalization and comparison helpers |
| `build_ground_truth.py` | Utility | Groups source data by UPC to create truth CSVs |

---

## Key Design Decisions

### Why a multi-stage pipeline instead of end-to-end SLM?

1. **Accuracy** -- Rules handle ~70% of attributes correctly before the SLM even runs. The SLM then refines the remaining ambiguities with full context.
2. **Efficiency** -- N-gram matching and rules are fast (seconds for 100 rows). Only Stage 4 (SLM inference) is slow (~8s/row). Pre-filtering means the SLM sees high-quality candidates rather than guessing from scratch.
3. **Debuggability** -- Each stage's output is inspectable. If a FLAVOUR is wrong, you can check: did the rule match? Did similarity return the right candidate? Did the SLM override incorrectly?
4. **Maintainability** -- Adding a new brand requires only a dictionary CSV update. Adding a new flavour pattern requires one regex in `rule_prefills.py`. No model retraining needed for dictionary changes.

### Why dictionary coverage scoring instead of cosine similarity?

Dictionary coverage (`|intersection| / |dict_tokens|`) measures how much of the dictionary entry appears in the row, not how much of the row appears in the dictionary. This is critical because rows are noisy and verbose (20+ tokens), while dictionary entries are concise (2-5 tokens). A short dictionary entry like `"SLURM"` fully covered by the row should score 1.0 regardless of how many other tokens the row has.

### Why valid-value snapping?

The SLM occasionally generates values that are close but not exactly matching the canonical list (e.g., `"BOTTLE-PET"` vs `"BOTTLE - PET"`). Snapping via fuzzy match (cutoff 70) corrects these without rejecting otherwise correct answers.

### Why prefill fallback?

The SLM generates JSON with 10 fields. Occasionally, the response is truncated (hitting the 256-token limit) or a field is left null. The prefill fallback ensures deterministic values (especially COUNT and SIZE_ML, which are almost always extractable by regex) are never lost.
