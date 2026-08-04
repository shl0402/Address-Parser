# Hong Kong address dataset generator review

## Outcome

The three scripts were reviewed as one pipeline and updated. The main invariant
is now: **a non-empty training target must be derived from text that is visibly
present in the final input after all augmentation**.

## 1. `build_hk_address_dataset (2)(1).py`

### Findings and applied changes

| Finding | Risk | Applied change |
| --- | --- | --- |
| The write/audit block was outside the `variant_index` loop. All variants were built, but only the last variant was written. | Dataset size and augmentation distribution were wrong. | Moved validation, writing, and audit observation inside the variant loop. |
| Street/village numbers shared the `BUILDING_NUMBER` label, while payload sanitation checked only whether that label appeared anywhere. | A visible village number could keep an omitted street number in metadata, or vice versa. | Added `component_kind` to each entity and made sanitation check both semantic chunk and label. |
| Removing only `BUILDING_NUMBER` left an unlabelled space at the edge of English street chunks. | Stray whitespace and malformed-looking addresses. | Added edge-separator trimming after atom-level omission. |
| The CLI option `--synthetic-3d-rate` was validated and audited but ignored; generation used hard-coded `0.80`/`0.95` values. | The command-line configuration did not describe the generated dataset. | Wired generation to `args.synthetic_3d_rate` for both monolingual and mixed rows. |
| Several dataset-level augmentation probabilities were hard-coded (`0.15`, `0.60`, `0.28`, `0.10`). | Rebalancing required code edits and reports could not reproduce the true configuration. | Added CLI options for Chinese separator noise, synthetic sub-districts, district suffix noise, Chinese region variants, and canonical/augmented floor-unit fusion. All are audited. |
| A random plausible sub-district was added to almost every eligible row, including canonical rows. A district is not enough to infer an address's true locality. | Geographically false combinations and strong hard-coded locality bias. | Synthetic sub-district generation is now opt-in with `--synthetic-subdistrict-rate`; the default is `0.0`, and it only applies to augmented variants. Source `LocationName` values are still used. |
| Source-row filtering inspected only the first usable language. | A row could be discarded when the field existed in its parallel language. | A field is now considered missing only when it is absent in every usable language. |
| Probability and English case-weight arguments were incompletely validated. | Invalid values silently produced unexpected distributions. | Validate all probabilities, filter rates, and case weights before writing output. |
| `leading_zero_unit` appeared twice in the standard 3D weight table. | Accidental double weighting. | Consolidated it into one explicit weight. |
| The default input path was the script directory rather than the documented `geojson/` folder. | Accidental discovery of unrelated JSON files. | Default is now `<script directory>/geojson`. |

### Important behavior retained

- Street number and street name are a single `street` chunk, so every reorder
  operation moves them together.
- Village number and village name are similarly grouped.
- The canonical/source payload is audit metadata. Character-span entities and
  `observed_components` remain the source for training targets.

## 2. `extract_compact_hk_address_jsonl(2).py`

### Findings and applied changes

| Finding | Risk | Applied change |
| --- | --- | --- |
| `street_phrases()` correctly implemented adjacent-number grouping but was never called. `--include-street-number` did not affect normal entity rows. | The option was effectively ignored and street numbers did not stay in `street_name`. | Entity extraction now uses `street_phrases()`; fallback extraction uses `fallback_street()`. |
| Adjacency alone cannot always distinguish a street number from a Chinese village number at the end of the preceding chunk. | A village number could be attached to a reordered street. | Prefer the new entity `component_kind`; fall back to adjacency only for older datasets without it. |
| Duplicate sub-district removal used global `random.random()`. | The same input generated different outputs on different runs. | Duplicate removal is deterministic. `--keep-duplicate-sub-district` explicitly controls whether duplicates are retained. |
| Entity text was whitespace-normalized before checking its offsets. | Valid spans containing intentional whitespace could be rejected. | Validate the stored entity text exactly against `text[start:end]`. |
| Region-code normalization handled only a small subset of punctuation and Chinese variants. | Equivalent visible forms produced inconsistent region codes. | Normalize dotted/spaced HK/NT forms and common Traditional/Simplified region variants. Text mode still preserves visible input text. |

## 3. `village_dataset_genV3(1).py`

### Findings and applied changes

| Finding | Risk | Applied change |
| --- | --- | --- |
| The output target was created before omission, abbreviation, casing, typo, fusion, punctuation, and code-switch operations. Only a few later changes manually updated it. | The output frequently contained values absent from the input. | Render labelled final chunks, then derive every output field from those final entities. Added a runtime visibility assertion. |
| House/DD-lot number and village name were separate chunks; `building_number` had no canonical-order priority. | The number could move far away from its village, even in canonical output. | Combined number and village name in one compound `village` chunk while retaining separate labels. Atom-level number omission is still supported. |
| Generation used one mutable RNG across all villages. | A source-order change altered every later generated row. | Reset generation RNG deterministically from seed, group ID, and variant index. Repeated runs are byte-identical. |
| Region inference deliberately generated a wrong HK/KLN value for some New Territories rows. | False geographic labels. | Infer canonical region deterministically from official district. Missing district produces an empty region instead of guessed geography; abbreviations/visible variants are separate augmentation steps. |
| Synthetic sub-districts were added at a hard-coded 90% rate. | False locality labels and excessive bias. | Disabled by default and exposed through `--synthetic-subdistrict-rate`. Real recognised-village sub-districts are retained. |
| Missing commas concatenated `YUEN LONG` with `POK FU LAM`, `元朗` with `薄扶林`, and `地區` with `VILLAGES IN`. | Silent parser corruption. | Restored the missing commas and added an adjacent-string check during verification. |
| Recognised-village section headers did not terminate the preceding record. | Headers and the next section leaked into village names. | Reworked block parsing so headers are case-insensitive record boundaries. Header contamination fell to zero in the embedded data. |
| Deduplication used only English village name. | Same-name villages in different districts could be merged. | Deduplicate by normalized bilingual name plus district and prefer the more complete duplicate. |
| Source text, output location, language mix, 3D rate, block/phase rate, and several augmentation rates were embedded in logic. | Updating sources or distributions required editing code. | Added external UTF-8 source-text options, script-relative output, language weights, output layout, development limit, and CLI-controlled dataset-level rates. Embedded source text remains a backward-compatible fallback. |
| Output was overwritten directly and the parent directory was assumed to exist. | Partial files on failure and fragile default path. | Validate overwrite intent, create the parent, write a temporary file, and atomically publish on success. |
| Simplified Chinese silently used a small manual replacement table when OpenCC was missing. | Partially converted, mixed-script labels. | Require OpenCC when Simplified Chinese is enabled; `--no-simplified` is the explicit fallback. |

## Verification performed

- All three files compile with `python -m py_compile`.
- All three `--help` entry points run.
- A one-feature bilingual ALS fixture with `--variants-per-address 3` writes six
  rows (two languages × three variants), proving the variant-loop fix.
- An aggressive urban stress run generated 40 rows with omission, reorder,
  typo, region noise, Chinese separator noise, and floor/unit fusion at maximum
  rates. Every non-empty compact target was found in its input.
- An aggressive village stress run generated 50 rows with the same invariant;
  every non-empty target was found in its input.
- Repeating the village run with the same seed produced the same SHA-256 hash.
- A focused Chinese test proved that a village `BUILDING_NUMBER` is not attached
  to an adjacent `STREET_NAME` when semantic chunk metadata is available.
- Invalid probability tests correctly exit with an error instead of generating
  a misleading dataset.

## Recommended usage notes

- Keep `--synthetic-subdistrict-rate 0` unless geographically false but
  syntactically plausible sub-districts are an intentional experiment.
- Synthetic floor/unit and village house/lot values are parser training data,
  not verified premises.
- For a district target such as `Lam Tin`, use
  `--district-source sub_district-then-official` in the compact extractor. For
  official District Council districts, retain the default `official` mode.
- Use `--region-format text` to preserve visible full-text/messy region forms,
  or `code` to normalize them to HK/KLN/NT.
- Install `opencc-python-reimplemented` and `tqdm` for the default Simplified
  Chinese and progress workflows. Use `--no-simplified --no-progress` only when
  those features are intentionally disabled.
