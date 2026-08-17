# Hong Kong Address Parser

A sequence-labelling model that takes a single-line Hong Kong address and parses it into structured 3-D address components:

`flat` · `floor` · `block` · `phase` · `building` · `estate` · `street` · `buildingnum` · `village` · `sub_district` · `district` · `region`

After parsing, the model can also reassemble the components into the conventional two-line postal format (Line 1 / Line 2).

---

## 1. Data Sources

### Urban addresses
Download the official GeoJSON from the Hong Kong Government:

- [Address Lookup Service – GeoJSON](https://data.gov.hk/en-data/dataset/hk-dpo-als_01-als/resource/92eed8bd-9e6f-4aac-9c9a-5ea6f5dbb4b8)

Place the downloaded files into a folder named `geojson/`.

### Rural / Village addresses
Sample data is generated from two Lands Department sources:

1. [List of Recognised Villages (rv0909.pdf)](https://www.landsd.gov.hk/doc/en/small-house/rv0909.pdf)  
2. [Place Name Gazetteer (more complete list)](https://www.landsd.gov.hk/doc/en/mapping/geographical-place-naming/Place_Name_Gazetteer.pdf)

---

## 2. Dataset Preparation

### 2.1 Urban data

```bash
# Build the initial dataset from GeoJSON
python build_hk_address_dataset_edit.py \
    --input-dir geojson \
    --output-dir hk_address_dataset

# Convert to compact JSONL format (region as plain text)
python extract_compact_hk_address_jsonl_edit.py \
    --input hk_address_dataset/train.jsonl \
    --output compact_train.jsonl \
    --region-format text

python extract_compact_hk_address_jsonl_edit.py \
    --input hk_address_dataset/test.jsonl \
    --output compact_test.jsonl \
    --region-format text

python extract_compact_hk_address_jsonl_edit.py \
    --input hk_address_dataset/validation.jsonl \
    --output compact_validation.jsonl \
    --region-format text
```

### 2.2 Rural / Village data

```bash
python village_dataset_gen.py      # version 1
python village_dataset_genV2.py    # version 2 (more villages)
python village_dataset_genV3_edit.py    # version 3 (ammended and merged v1 and v2)
```

### 2.3 Merge everything

```bash
python merge_data.py
```

This produces the merged data files:

- `train.jsonl`
- `validation.jsonl`
- `test.jsonl`

### 2.4 Data Augmentation and Cleaning

```bash
python clean_address_jsonl.py
```

This produces the final training-ready files:

- `train_cleaned.jsonl`
- `validation_cleaned.jsonl`
- `test_cleaned.jsonl`

---

## 3. Training

Three model variants are provided:

| Notebook                    | Model                     |
|-----------------------------|---------------------------|
| `train_bilstmV7.ipynb`        | [BiLSTM](https://huggingface.co/shl0402/bilstm_v7)                    |
| `train_bert.ipynb`          | XLM-RoBERTa-base          |
| `train_bert_largeV4.ipynb`  | [XLM-RoBERTa-large](https://huggingface.co/shl0402/large-roberta-address-parser)         |

Simply open the desired notebook and run first cell to train.  
The notebooks expect the three JSONL files produced by in the above to be present in the working directory.

---

## 4. Inference & Evaluation

Use `bilstm_parser.py`.

The key function is `split_address(...)`, which:

Re-assembles them into the conventional two-line Hong Kong postal format (Line 1 + Line 2).

---

## License

Please respect the terms of use of the original government data sources.
