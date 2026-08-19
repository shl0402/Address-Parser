# Hong Kong Address Parser

A sequence-labelling pipeline that takes a **single-line Hong Kong address**
(English, Chinese, or mixed) and:

1. Tags every token as a 3-D address field (`flat`, `floor`, `block`, …).
2. Rebuilds the conventional **two-line postal split** (Line 1 / Line 2).
3. Returns per-field confidence, a split-logic confidence, and a
   `[SUCCESS]` / `[WARNING]` / `[ALERT]` status.

Two model backends share the same output contract:

| Backend | File | Architecture | Typical use |
|---|---|---|---|
| BiLSTM | `bilstm_parser.py` | Word + char-CNN + RoPE + BiLSTM + CRF | Fast, small, no Hugging Face download |
| BERT | `bert_parser.py` | XLM-RoBERTa-large + constrained CRF | Stronger on messy / mixed-language input |

If you have never touched this repo, read this file top to bottom once.
After that, use **§8 What to change** as the index whenever you edit logic.

---

## Table of contents

1. [Mental model](#1-mental-model)
2. [Repository layout](#2-repository-layout)
3. [Python environment & `requirements.txt`](#3-python-environment--requirementstxt)
4. [Dataset preparation](#4-dataset-preparation)
5. [JSONL format the trainers expect](#5-jsonl-format-the-trainers-expect)
6. [Training](#6-training)
7. [Inference](#7-inference)
8. [What to change (logic map)](#8-what-to-change-logic-map)
9. [Split logic in detail](#9-split-logic-in-detail)
10. [Tokenization, labels, confidence, status](#10-tokenization-labels-confidence-status)
11. [Evaluation](#11-evaluation)
12. [Checkpoints & file contracts](#12-checkpoints--file-contracts)
13. [Known pitfalls](#13-known-pitfalls)
14. [License / data terms](#14-license--data-terms)

---

## 1. Mental model

```
raw address string
        │
        ▼
  tokenizer  ──►  sequence of tokens (or subwords)
        │
        ▼
  NER model + CRF  ──►  BIO tags per token
        │                 B-UNIT  I-FLOOR  B-STREET_NAME  …
        │
        ▼
  char-level projection  ──►  tag + confidence per character
        │
        ▼
  entity grouping  ──►  {flat, floor, block, building_name, …}
        │
        ▼
  _split_address()  ──►  Line 1, Line 2, split_conf, used_keys, is_reversed
        │
        ▼
  status string  ──►  [SUCCESS] | [WARNING]: … | [ALERT]: …
```

The NER model only answers “what is each span?”. The **two-line postal
format is a deterministic rule** in `_split_address()`, not something the
network learns. If Line 1 / Line 2 look wrong but the tags look right,
edit the split function — do not retrain.

Hong Kong convention (simplified):

- **English** addresses: micro fields first (flat / floor / block / building),
  then street / district / region.
- **Chinese** addresses: the opposite — district / street first, then
  building / floor / flat.

The parser detects Chinese with the CJK range `\u4e00–\u9fff` and swaps
which of `{micro, macro}` becomes Line 1.

---

## 2. Repository layout

```
.
├── README.md                          ← this file
├── requirements.txt                   ← CPU (inference / notebooks)
├── requirements-gpu.txt               ← CUDA (training)
├── requirements-data.txt              ← extras for dataset scripts
│
├── bert_parser.py                     ← production inference (XLM-R + CRF)
├── bilstm_parser.py                   ← production inference (BiLSTM-CNN-CRF)
│
├── train_bert_largeV4.ipynb           ← train XLM-RoBERTa-large + CRF
├── train_bilstmV7.ipynb               ← BiLSTM grid search (108 trials)
│
├── dataset_preparation/               ← scripts that build the JSONL
│   ├── build_hk_address_dataset_edit.py
│   ├── extract_compact_hk_address_jsonl_edit.py
│   ├── village_dataset_gen.py         ← v1 (recognised villages)
│   ├── village_dataset_genV2.py       ← v2 (gazetteer, more villages)
│   ├── village_dataset_genV3_edit.py  ← v3 = amended merge of v1+v2  ← use this
│   ├── merge_data.py
│   └── clean_address_jsonl.py
│
├── geojson/                           ← official ALS GeoJSON (you download)
├── data2/                             ← JSONL the trainers actually read
│   ├── train_cleaned.jsonl
│   ├── validation_cleaned.jsonl
│   ├── test_cleaned.jsonl
│   └── address_dataset.jsonl          ← external Line1/Line2 eval set
│
├── xlm_roberta_large_crfV4/           ← best BERT weights (created by training)
├── xlm_roberta_large_checkpointV4/    ← BERT resume checkpoint
├── xlm_roberta_large_cacheV4/         ← tokenized HF datasets cache
└── bilstm_crf_modelV7_search/         ← one folder per BiLSTM trial
    └── trial_XXX_…/
        ├── best_by_logic.bin
        ├── vocabs.json
        └── checkpoint_epoch_XX.pt
```

Paths inside the notebooks are **hardcoded** (`data2/…`, `./xlm_roberta_large_crfV4`,
`./bilstm_crf_modelV7_search`). Either keep that layout or change the constants
at the top of the first cell.

---

## 3. Python environment & `requirements.txt`

### Verdict on the original file

The original `requirements.txt` was **not enough**.

| Item | Original | Problem |
|---|---|---|
| `torch==2.7.1+cpu` | present | CPU-only. Training XLM-RoBERTa-large on CPU is not practical. Notebooks call `nvidia-smi`. |
| `sentencepiece` | **missing** | XLM-RoBERTa’s tokenizer is SentencePiece. `AutoTokenizer.from_pretrained("xlm-roberta-large")` fails without it. |
| `protobuf` | **missing** | Required by SentencePiece on many setups. |
| `numpy` | **missing** | Parsers call `.cpu().numpy()` on offset maps. Usually pulled in by torch, but pin it. |
| `safetensors`, `tokenizers`, `huggingface_hub` | **missing** | Required (or strongly expected) by `transformers` 5.x. |
| Jupyter | only `ipywidgets` | You cannot *run* the `.ipynb` files without `ipykernel` + `notebook`/`jupyter`. |
| Dataset scripts | nothing | GeoJSON / village-PDF builders need `pandas`, `geojson`/`shapely`, `pypdf`/`pdfplumber`. |

What *was* correct:

- `pytorch-crf==0.7.2` — this is the package that provides `from torchcrf import CRF`.
- `transformers==5.14.0` and `datasets==5.0.0` — valid 2026 pins.
- `tqdm==4.68.4`.
- `--extra-index-url` for the PyTorch CPU wheel index.

### Install

```bash
# 1. Create an isolated env (Python 3.12 matches the notebooks)
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -U pip

# 2a. Inference / CPU machine
pip install -r requirements.txt

# 2b. Training on an NVIDIA GPU  (check `nvidia-smi` first)
pip install -r requirements-gpu.txt

# 2c. Only if you will rebuild the JSONL from GeoJSON / PDFs
pip install -r requirements-data.txt
```

GPU pin: `requirements-gpu.txt` defaults to **CUDA 12.6** (`torch==2.7.1+cu126`).
If `nvidia-smi` shows a 12.8-capable driver, swap the extra-index-url and the
torch line to `cu128` (comments in that file).

First BERT run also downloads `xlm-roberta-large` from Hugging Face
(~2.2 GB). Set `HF_HOME` if you want the cache on a bigger disk.

VRAM ballpark:

- BiLSTM V7 (`hidden_dim=320`, batch 256): a few GB.
- XLM-RoBERTa-large (`BATCH_SIZE=16`, `ACCUMULATION_STEPS=8`, `MAX_LEN=128`):
  **~16–24 GB**. Drop `BATCH_SIZE` if you OOM; keep the product
  `BATCH_SIZE * ACCUMULATION_STEPS` around 128 to preserve the effective batch.

---

## 4. Dataset preparation

Training does **not** read GeoJSON. It reads three cleaned JSONL files.
Build them once, then you can iterate on models without touching this section.

### 4.1 Urban addresses (Address Lookup Service)

Download the official GeoJSON:

- [Address Lookup Service – GeoJSON](https://data.gov.hk/en-data/dataset/hk-dpo-als_01-als/resource/92eed8bd-9e6f-4aac-9c9a-5ea6f5dbb4b8)

Put the files in a folder named `geojson/`.

```bash
python build_hk_address_dataset_edit.py \
    --input-dir geojson \
    --output-dir hk_address_dataset

# Compact JSONL, region stored as plain text (not a nested object)
python extract_compact_hk_address_jsonl_edit.py \
    --input  hk_address_dataset/train.jsonl \
    --output compact_train.jsonl \
    --region-format text

python extract_compact_hk_address_jsonl_edit.py \
    --input  hk_address_dataset/validation.jsonl \
    --output compact_validation.jsonl \
    --region-format text

python extract_compact_hk_address_jsonl_edit.py \
    --input  hk_address_dataset/test.jsonl \
    --output compact_test.jsonl \
    --region-format text
```

### 4.2 Rural / village addresses

Sources:

1. [List of Recognised Villages (rv0909.pdf)](https://www.landsd.gov.hk/doc/en/small-house/rv0909.pdf)
2. [Place Name Gazetteer](https://www.landsd.gov.hk/doc/en/mapping/geographical-place-naming/Place_Name_Gazetteer.pdf)

```bash
python village_dataset_gen.py          # v1
python village_dataset_genV2.py        # v2, more villages
python village_dataset_genV3_edit.py   # v3 — amended merge of v1 + v2. Use this.
```

### 4.3 Merge + clean

```bash
python merge_data.py          # → train.jsonl / validation.jsonl / test.jsonl
python clean_address_jsonl.py # → train_cleaned.jsonl / validation_cleaned.jsonl / test_cleaned.jsonl
```

Copy (or symlink) the three `*_cleaned.jsonl` files into `data2/`, which is
where both training notebooks look. Also place the external Line1/Line2 set
at `data2/address_dataset.jsonl` (used as the BiLSTM *selection* metric and
as BERT’s *monitoring* metric).

Approximate sizes from a previous run: Train ~314k, Val ~39k, external test
~1.8k.

---

## 5. JSONL format the trainers expect

One JSON object per line.

**Tagged form** (what `train_cleaned.jsonl` looks like):

```json
{
  "input": "Flat A, 5/F, Block 2, Happy Mansion, 28 Lockhart Road, Wan Chai, Hong Kong",
  "output": {
    "line1": {
      "flat": "Flat A",
      "floor": "5/F",
      "block": "Block 2",
      "building_name": "Happy Mansion"
    },
    "line2": {
      "building_number": "28",
      "street_name": "Lockhart Road",
      "district": "Wan Chai",
      "region": "Hong Kong"
    }
  }
}
```

A flat `output` (no `line1`/`line2` wrappers) is also accepted — both
`reconstruct_char_labels()` functions flatten either shape.

**Line-string form** (what `address_dataset.jsonl` often looks like):

```json
{
  "input": "Flat A, 5/F, Happy Mansion, 28 Lockhart Road, Wan Chai, Hong Kong",
  "output": {
    "line1": "Flat A, 5/F, Happy Mansion",
    "line2": "28 Lockhart Road, Wan Chai, Hong Kong"
  }
}
```

External eval also accepts a TXT file:

```
input_address | GT_line1 | GT_line2
```

Gold spans are projected onto the input by **substring search**
(`reconstruct_char_labels`). Longer values are applied first so
`Lockhart Road` is not eaten by `Road`. Matching is case-insensitive in
the BERT trainer and case-sensitive in the BiLSTM trainer — keep gold
strings as they appear in `input`.

If a gold value is **not a contiguous substring of `input`**, that sample
is treated as corrupted and skipped at eval time.

---

## 6. Training

Open the notebook and run the **first cell**. That cell is the full
training script (`if __name__ == "__main__": main()`). Later cells are
evaluation helpers; they are not required to produce weights.

### 6.1 BiLSTM — `train_bilstmV7.ipynb`

What it does:

- Builds word / char vocabs from `data2/train_cleaned.jsonl`.
- Runs a **108-trial grid** (lr × dropout × weight_decay × hidden_dim ×
  num_layers). `char_cnn_filters` and `batch_size` are currently fixed.
- Primary model-selection metric: **Test Logic Acc** on
  `data2/address_dataset.jsonl` (Line 1 / Line 2 string match, not entity F1).
- Early stop: patience 3 after epoch 8, on Test Logic Acc.
- Each trial writes its own folder under `./bilstm_crf_modelV7_search/`.

Constants you actually edit (top of cell 0):

```python
TRAIN_FILE = "data2/train_cleaned.jsonl"
VAL_FILE   = "data2/validation_cleaned.jsonl"
TEST_FILE  = "data2/address_dataset.jsonl"
BASE_OUTPUT_DIR = "./bilstm_crf_modelV7_search"

GRID = {
    "learning_rate":    [1.5e-4, 2.5e-4, 3.5e-4],
    "dropout":          [0.40, 0.50, 0.60],
    "weight_decay":     [1e-5, 3e-5, 1e-4],
    "hidden_dim":       [256, 320],
    "num_layers":       [1, 2],
    "char_cnn_filters": [100],     # keep even: word_dim(300)+filters must be even for RoPE
    "batch_size":       [256],
}
EPOCHS = 30
EARLY_STOP_PATIENCE = 3
MIN_EPOCHS_BEFORE_STOP = 8
```

Architecture knobs that **must match inference** (`bilstm_parser.py`):

```python
WORD_EMBED_DIM = 300
CHAR_EMBED_DIM = 100
CHAR_CNN_FILTERS = 100          # parser constant CHAR_CNN_FILTERS
hidden_dim / num_layers / dropout   # passed into HKAddressParserBiLSTM(...)
```

RoPE is applied on `word_dim + cnn_filters`. That sum **must be even**.

Load the winner:

```python
from bilstm_parser import HKAddressParserBiLSTM
parser = HKAddressParserBiLSTM(
    model_path="bilstm_crf_modelV7_search/trial_XXX_…",  # folder with best_by_logic.bin + vocabs.json
    hidden_dim=320,   # must match that trial
    num_layers=2,
    dropout=0.4,
)
```

`HKAddressParserBiLSTM` prefers `best_by_logic.bin`, then falls back to
`pytorch_model.bin`.

### 6.2 BERT — `train_bert_largeV4.ipynb`

What it does:

- Reconstructs char-level BIO labels, aligns them onto XLM-R subwords via
  `offset_mapping`, sanitises illegal `I-X` (promotes to `B-X`).
- Caches the tokenized Hugging Face datasets under
  `./xlm_roberta_large_cacheV4/`. Delete that folder if you change data or
  the tokenizer.
- Trains XLM-RoBERTa-large + a **frozen BIO-constrained CRF**.
- Best checkpoint is chosen by **validation CRF loss** (external Line1/Line2
  accuracy is logged every 2000 steps but does **not** select the model).
- Resumes from `./xlm_roberta_large_checkpointV4/` if
  `training_state.pt` is present.
- Writes `checkpoint_epoch_XX/` under `./xlm_roberta_large_crfV4/` every epoch
  and copies the best weights to that folder’s root.

Constants (top of cell 0):

```python
MODEL_PATH = "xlm-roberta-large"
OUTPUT_DIR = "./xlm_roberta_large_crfV4"
CHECKPOINT_DIR = "./xlm_roberta_large_checkpointV4"
DATA_CACHE_DIR = "./xlm_roberta_large_cacheV4"

TRAIN_FILE = "data2/train_cleaned.jsonl"
VAL_FILE   = "data2/validation_cleaned.jsonl"
TEST_FILE  = "data2/test_cleaned.jsonl"
EXTERNAL_TEST_FILE = "data2/address_dataset.jsonl"

EPOCHS = 5
LEARNING_RATE = 8e-6          # encoder
# CRF + classifier use a separate group at 1e-3
WEIGHT_DECAY = 0.05
BATCH_SIZE = 16
ACCUMULATION_STEPS = 8        # effective batch = 128
MAX_LEN = 128
EVAL_EVERY_STEPS = 2000
SAVE_STEPS = 300
```

Load:

```python
from bert_parser import HKAddressParserBERT
parser = HKAddressParserBERT(model_path="xlm_roberta_large_crfV4")
```

The folder must contain `config.json`, tokenizer files
(`sentencepiece.bpe.model`, `tokenizer_config.json`, …) and
`pytorch_model.bin`.

---

## 7. Inference

Both parsers share the same call shape and the same 8-tuple result.

```python
from bert_parser import HKAddressParserBERT
# from bilstm_parser import HKAddressParserBiLSTM

parser = HKAddressParserBERT("xlm_roberta_large_crfV4", conf_threshold=0.50)

# one address, optionally already split into two postal lines
addr, line1, line2, tags, confs, split_conf, used_keys, status = parser.parse(
    "Flat A, 5/F, Happy Mansion",
    "28 Lockhart Road, Wan Chai, Hong Kong",
)

# many addresses
results = parser.parse_batch(
    [("Flat A, UG, Happy Mansion, 28 Lockhart Road, Wan Chai, Hong Kong", ""),
     ("長洲東灣東堤小築彌敦道一百二十三12座H室5樓", "")],
    batch_size=32,
)
```

Return tuple:

| Index | Name | Meaning |
|---|---|---|
| 0 | `address_str` | The string actually tagged (parts joined with language-aware punctuation) |
| 1 | `line1` | Micro (EN) or macro (ZH) |
| 2 | `line2` | The other group |
| 3 | `tags` | `dict[str, str]` — joined surface forms, standard keys |
| 4 | `confs` | `dict[str, float]` — mean softmax of the CRF emission at the chosen tag |
| 5 | `split_conf` | Product of confidences of the keys used to decide the split |
| 6 | `used_keys` | Those keys |
| 7 | `status` | `[SUCCESS]` / `[WARNING]: …` / `[ALERT]: …` (can concatenate) |

Two-part join rules inside `parse_batch`:

- Empty → `EMPTY_INPUT`.
- One part → used as-is.
- Two parts, Chinese → `" ".join(parts)`.
- Two parts, English → `"{p1}, {p2}"` unless `p1` already ends with `,`.

GPU pick: `_get_emptiest_gpu_safely()` prefers a GPU with util < 30% and
the most free VRAM; falls back to CPU if CUDA is absent.

---

## 8. What to change (logic map)

**There is no single source of truth.** `_split_address`, the field↔tag
map, the tokenizer regex, and `ALL_FIELDS` are copy-pasted across:

| Location | Copies |
|---|---|
| `bert_parser.py` | production |
| `bilstm_parser.py` | production |
| `train_bert_largeV4.ipynb` | cells 0, 1, 2 |
| `train_bilstmV7.ipynb` | cells 0, 1, 2, 3, 4 |

If you change a rule, **grep the symbol and update every copy**, or the
train-time metric will disagree with the parser you ship.

### 8.1 Add / rename an entity field

Example: you want a `lot` field.

1. Add `"lot": "LOT"` to `field_to_tag` in `reconstruct_char_labels`
   (both trainers).
2. Add `"LOT"` to `tag_list` (BIO `B-LOT` / `I-LOT` are generated from it).
3. Add `"LOT": "lot"` to `mapping` in `_map_extracted_to_standard_keys`
   **and** inside `_split_address`.
4. Add `"lot"` to `ALL_FIELDS` (eval).
5. Decide whether `lot` is **micro** (Line 1 in English) or **macro**
   (Line 2 in English) and add a branch in `_split_address` (see §9).
6. Retrain. Old checkpoints do not know the new tag id.

### 8.2 Change the two-line split

Edit `_split_address` (see §9). This is the function to touch if:

- Block + building are landing on the wrong line.
- Village numbers should stay with the village name (they already do).
- You want phase to travel with the estate even when a building name is
  also present (today, `has_block` / `has_bldg` short-circuit before
  `has_est`).

**Do not retrain** for a pure split-rule change. Re-run the eval cells.

### 8.3 Change tokenization (BiLSTM only)

Training (`tokenize_and_align`) and inference (`_tokenize_text`) **must
use the same regex**. Today they do **not**:

```python
# training (train_bilstmV7.ipynb)
r'[a-zA-Z]+|[0-9]+|[\u4e00-\u9fff]|[^\s]'
# → "21A" becomes ["21", "A"]

# inference (bilstm_parser.py)
r'[a-zA-Z0-9]+|[\u4e00-\u9fff]|[^\s]'
# → "21A" stays ["21A"]
```

If you change one, change the other and rebuild `vocabs.json` (i.e.
retrain, because the word vocab changes). BERT is not affected — it uses
the XLM-R SentencePiece tokenizer + `offset_mapping`.

BERT *entity regrouping* after decoding still uses a similar regex in
`parse_batch` (`[a-zA-Z0-9]+|…`). That only affects how tags are *printed*,
not how the net is trained.

### 8.4 Change the confidence warning threshold

```python
HKAddressParserBERT(model_path, conf_threshold=0.50)
HKAddressParserBiLSTM(model_path, conf_threshold=0.50)
```

`split_conf` is the **product** of the confidences of `logic_keys_used`.
A 0.50 threshold is therefore much stricter than “every field ≥ 0.50”
when several keys are multiplied.

If `logic_keys_used` is empty, the product is taken over every field
except `district` / `region` / `sub_district`.

### 8.5 Change Chinese vs English line order

```python
is_chinese = any("\u4e00" <= char <= "\u9fff" for char in original_input)
line1 = macro_string if is_chinese else micro_string
line2 = micro_string if is_chinese else macro_string
```

A single CJK character in an otherwise English address flips the order.
If that is too aggressive, change this predicate in `_split_address`.

The same flag also picks the default group for untagged (`O`) tokens
(`last_valid`): English defaults to `micro`, Chinese to `macro`. That is
how punctuation between fields inherits a group.

### 8.6 Change BIO hard constraints (BERT CRF)

`BertCRFForTokenClassification._set_bio_constraints` fills every
transition with `-1e4` then allows:

- `O → O`, `O → B-*`
- `B-X → I-X`, `B-X → O`, `B-X → B-*`
- `I-X → I-X`, `I-X → O`, `I-X → B-*`
- start: `O` or any `B-*`
- end: `O`, any `B-*`, any `I-*`

Illegal sequences such as `O → I-X` or `B-UNIT → I-FLOOR` cannot be
decoded. Transitions are then **frozen** (`requires_grad_(False)`), so
the CRF is a hard grammar, not a learned one.

The BiLSTM CRF is **unconstrained** (learned transitions).

### 8.7 Change hyperparameters without a code hunt

| Want | Where |
|---|---|
| BERT lr / epochs / max length | top of `train_bert_largeV4.ipynb` cell 0 |
| BERT encoder vs CRF lr | `AdamW` param groups in `main()` — encoder `LEARNING_RATE`, CRF/classifier `1e-3` |
| Dropout on BERT head | `nn.Dropout(0.2)` inside `BertCRFForTokenClassification` |
| BiLSTM grid | `GRID` dict, cell 0 of `train_bilstmV7.ipynb` |
| BiLSTM inference architecture | constructor args of `HKAddressParserBiLSTM` **must equal the trial** |
| RoPE base / max length | `ROPE_BASE`, `MAX_SEQ_LEN_ROPE` in both trainer and `bilstm_parser.py` |

### 8.8 Change how two input parts are joined

`parse_batch` in each parser, the `for p1, p2 in batch_pairs` block.
Training never sees two-part input — it trains on the already-joined
`input` field of the JSONL.

---

## 9. Split logic in detail

`_split_address(extracted_labels, original_input, parsed_entities, conf_mapped)`
is the entire postal-format policy.

### 9.1 Which keys are “micro” (Line 1 in English)

Always micro if present: `flat`, `floor`.

Then **exactly one** of these branches fires (if / elif):

```
has_block?
    micro += block
    if building_name is present:
        BERT parser:      also add building_name   ← always
        BiLSTM parser:    add building_name only when estate_name is ALSO present
                          (elif has_bldg and not has_est: pass)
has_building_name?   (no block)
    micro += building_name
has_estate_name?
    micro += estate_name
    if phase: micro += phase
has_village_name?
    micro += village_name
    if building_number: micro += building_number
has_street_name?
    micro += street_name
    if building_number: micro += building_number
has_building_number?
    micro += building_number
```

Everything tagged that is **not** in that micro set becomes **macro**
(Line 2 in English): typically `street_name`, `building_number`,
`sub_district`, `district`, `region`, leftover `estate_name` / `phase` /
`village_name`.

> **Divergence:** production `bert_parser.py` and `bilstm_parser.py`
> disagree on `block + building_name` without an estate. BERT puts the
> building name on Line 1; BiLSTM leaves it on Line 2. The BERT *training*
> notebook follows the BERT parser; the BiLSTM *training* notebook follows
> the BiLSTM parser. Unify these before you compare backends.

### 9.2 How strings are actually built

1. Map every entity to `{micro, macro, O}`.
2. Fill `O` (punctuation, “No.”, extra words) with the previous non-O
   group, or with the language default if the address starts with `O`.
3. Concatenate tokens **in user order**. There is no
   `reorder_segments()` anymore — an older version moved `building_number`
   next to the street/village; that was removed on purpose so the output
   stays in typed order.
4. Strip leading/trailing `, / \ - ;` and CJK punctuation; collapse spaces.
5. Swap micro/macro into Line 1 / Line 2 based on `is_chinese`.
6. `is_reversed` (inference parsers only): after stripping punctuation,
   `line1+line2` must equal the original string. If the model interleaved
   micro and macro spans (e.g. village then block then street then floor),
   concatenation reshuffles characters and this flag fires
   `[ALERT]: reversed order after parsing`.

### 9.3 Worked English example

Input: `27LD, Block 5, Hemera, Lohas Park, Lohas Road 1, TKO, HK`

Hypothetical tags:

| span | field |
|---|---|
| 27LD | flat |
| Block 5 | block |
| Hemera | building_name |
| Lohas Park | estate_name |
| Lohas Road | street_name |
| 1 | building_number |
| TKO | district |
| HK | region |

`has_block` is true, and both building and estate exist, so micro =
`{flat, floor, block, building_name}`. Line 1 ≈ `27LD, Block 5, Hemera`.
Line 2 ≈ `Lohas Park, Lohas Road 1, TKO, HK`.

### 9.4 Worked Chinese example

Input: `長洲東灣東堤小築12座H室5樓`

Micro still = flat/floor/block/building; because `is_chinese`, those
become **Line 2**. Line 1 is the leftover village / district / street
macro string.

---

## 10. Tokenization, labels, confidence, status

### Label set (25 tags)

```
O
B-UNIT          I-UNIT           ← mapped to key "flat"
B-FLOOR         I-FLOOR          ← "floor"
B-BLOCK         I-BLOCK          ← "block"
B-PHASE         I-PHASE          ← "phase"
B-BUILDING_NAME I-BUILDING_NAME  ← "building_name"
B-ESTATE_NAME   I-ESTATE_NAME    ← "estate_name"
B-VILLAGE_NAME  I-VILLAGE_NAME   ← "village_name"
B-BUILDING_NUMBER I-BUILDING_NUMBER ← "building_number"
B-STREET_NAME   I-STREET_NAME    ← "street_name"
B-SUB_DISTRICT  I-SUB_DISTRICT   ← "sub_district"
B-DISTRICT      I-DISTRICT       ← "district"
B-REGION        I-REGION         ← "region"
```

BERT stores these on `config.id2label` (JSON keys come back as strings;
the parser casts them to `int`). BiLSTM stores them in `vocabs.json`
under `t2i`.

### Confidence

For each token, `softmax(emissions)[token, chosen_tag]`. Entity
confidence is the mean over its tokens. `split_conf` multiplies the
entity confidences of `logic_keys_used` only — so a perfect district
tag cannot rescue a bad `flat` if `flat` was used to decide the split.

### Status construction (do not reorder if other modules parse it)

```
alerts first, then warnings, else [SUCCESS]

1 alert  → "[ALERT]: reversed order after parsing"
N alerts → "[ALERTS]: a; b"
1 warn   → "[WARNING]: low splitting confidence (0.12 < 0.5)"
N warns  → "[WARNINGS]: …"
```

An alert and a warning concatenate with no separator:
`[ALERT]: reversed order after parsing[WARNING]: low splitting confidence …`.

---

## 11. Evaluation

Each training notebook has extra cells that dump a log file and a
threshold table (`0 / 50 / 60 / 70 / 80 %` split_conf bins).

Two different “correctness” definitions exist — do not mix them:

| Notebook cell | Gold | Metric |
|---|---|---|
| BERT cell 2 | per-field tags in JSONL | field exact match (with floor+flat swap and building↔estate swap allowed) |
| BiLSTM cell 4 | `line1` / `line2` **strings** (TXT `input \| l1 \| l2`) | normalised string equality |
| BiLSTM training loop | `address_dataset.jsonl` line strings | Test Logic Acc (drives early stop) |
| BERT training loop | val CRF loss | drives “best” snapshot; external split acc is monitor-only |

Normalisation for string compare:

```python
re.sub(r'[\s,/\\\-;\.，。、；]+', '', text.lower())
```

Soft field-match exceptions (BERT eval cell):

- `floor`/`flat` may be swapped if `floor+flat` concatenates equally.
- `building_name` / `estate_name` may be swapped.

---

## 12. Checkpoints & file contracts

### BERT folder (`xlm_roberta_large_crfV4/`)

```
config.json                 # must contain id2label / label2id / hidden_size / num_labels
pytorch_model.bin           # full nn.Module state_dict (bert.* + classifier.* + crf.*)
tokenizer_config.json
sentencepiece.bpe.model
special_tokens_map.json
tokenizer.json              # optional, depends on transformers version
checkpoint_epoch_XX/        # same files + training_state.pt
training_metrics.log
metrics.jsonl
```

Resume folder `xlm_roberta_large_checkpointV4/` additionally has
`training_state.pt` (`epoch`, `step`, `global_step`, optimizer, scheduler,
`best_val_loss`).

`HKAddressParserBERT` also accepts a checkpoint dict that wraps weights
under `model_state_dict`.

### BiLSTM trial folder

```
vocabs.json           # {"w2i":..., "c2i":..., "t2i":...}  required
best_by_logic.bin     # preferred weights
pytorch_model.bin     # fallback name
checkpoint_epoch_XX.pt
training_metrics.log
metrics.jsonl
trial_result.json
```

`vocabs.json` is written only when a new best Test Logic Acc is hit.
If you copy weights without the matching vocab, every token becomes
`<UNK>` and quality collapses.

Grid search writes `bilstm_crf_modelV7_search/search_summary.json`
ranked by `best_test_logic`.

---

## 13. Known pitfalls

1. **Split-rule copies drift.** BERT vs BiLSTM already disagree on
   `block + building` without estate. Grep before editing.
2. **BiLSTM train/infer tokenizer mismatch** (`[a-zA-Z]+|[0-9]+` vs
   `[a-zA-Z0-9]+`). `21A`, `1M`, `27LD` are the canaries.
3. **`hidden_dim` / `num_layers` / `dropout` on BiLSTM inference must
   equal the trial** or `load_state_dict` silently skips tensors
   (the loader copies only matching shapes and prints `missing=`).
4. **Stale BERT cache.** Changing JSONL or `MAX_LEN` without deleting
   `xlm_roberta_large_cacheV4/` trains on the old tokenisation.
5. **Gold not a substring of input** → sample excluded from eval, which
   can inflate reported accuracy. Check the `[EXCLUDED: CORRUPTED DATA]`
   lines in the log.
6. **CPU `requirements.txt` + BERT training.** It will *run*, then crawl.
   Use `requirements-gpu.txt`.
7. **`mmap=True` in `bert_parser.py`’s `torch.load`.** Convenient for
   large bins; some torch/CPU combinations have been picky — drop `mmap`
   if load fails.
8. **Effective BERT batch.** `BATCH_SIZE=16` with `ACCUMULATION_STEPS=8`
   is an effective 128. Changing only one of them changes regularisation.
9. **`torch.backends.cudnn.enabled = False`** in the BiLSTM trainer (and
   some eval cells). Leave it unless you are chasing speed and have
   verified bitwise-stable results.
10. **First-time XLM-R download** needs network access to Hugging Face.
    For air-gapped machines, `huggingface-cli download xlm-roberta-large`
    elsewhere and point `MODEL_PATH` at the local folder.

---

## 14. License / data terms

Model code in this repo is for the project’s own use. The urban GeoJSON
and the Lands Department PDFs are government data — respect their terms
of use:

- [Address Lookup Service](https://data.gov.hk/en-data/dataset/hk-dpo-als_01-als/resource/92eed8bd-9e6f-4aac-9c9a-5ea6f5dbb4b8)
- [Recognised Villages](https://www.landsd.gov.hk/doc/en/small-house/rv0909.pdf)
- [Place Name Gazetteer](https://www.landsd.gov.hk/doc/en/mapping/geographical-place-naming/Place_Name_Gazetteer.pdf)
