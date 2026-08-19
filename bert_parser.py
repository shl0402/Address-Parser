#!/usr/bin/env python3
"""
HK Address Parser – Inference for BERT (XLM-RoBERTa-large + CRF)
Compatible with models trained by the provided BERT script.

Changes/Features:
- Identical input and output usage as HKAddressParserBiLSTM.
- Uses HuggingFace AutoTokenizer offset_mapping to perfectly map Subword tokens
  back to original string characters and extract exact confidence scores.
- Maintains the exact V7/V8 _split_address logic (including interleaving/reversal checks).
- Emits [ALERT] / [WARNING] statuses under the exact same logic structure.
"""

import os
import re
import json
import torch
import subprocess
import torch.nn as nn
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoConfig
from torchcrf import CRF


# ==========================================
# Model (XLM-RoBERTa + CRF with BIO constraints)
# ==========================================
class BertCRFForTokenClassification(nn.Module):
    def __init__(self, config, label_list):
        super().__init__()
        self.config = config
        self.label_list = label_list
        # We load the base architecture; weights will be injected via load_state_dict
        self.bert = AutoModel.from_config(config)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self._set_bio_constraints(label_list)

    def _set_bio_constraints(self, label_list):
        self.crf.transitions.data.fill_(-1e4)
        self.crf.start_transitions.data.fill_(-1e4)
        self.crf.end_transitions.data.fill_(-1e4)
        id2label = {i: l for i, l in enumerate(label_list)}
        label2id = {l: i for i, l in id2label.items()}

        o_id = label2id["O"]
        self.crf.transitions.data[o_id, o_id] = 0.0
        for tag in label_list:
            if tag.startswith("B-"):
                self.crf.transitions.data[o_id, label2id[tag]] = 0.0
        self.crf.start_transitions.data[o_id] = 0.0
        self.crf.end_transitions.data[o_id] = 0.0

        for tag in label_list:
            if tag == "O":
                continue
            tid = label2id[tag]
            if tag.startswith("B-"):
                itype = "I-" + tag[2:]
                if itype in label2id:
                    self.crf.transitions.data[tid, label2id[itype]] = 0.0
                self.crf.transitions.data[tid, o_id] = 0.0
                for other in label_list:
                    if other.startswith("B-"):
                        self.crf.transitions.data[tid, label2id[other]] = 0.0
                self.crf.start_transitions.data[tid] = 0.0
                self.crf.end_transitions.data[tid] = 0.0
            elif tag.startswith("I-"):
                self.crf.transitions.data[tid, tid] = 0.0
                self.crf.transitions.data[tid, o_id] = 0.0
                for other in label_list:
                    if other.startswith("B-"):
                        self.crf.transitions.data[tid, label2id[other]] = 0.0
                self.crf.end_transitions.data[tid] = 0.0

        self.crf.transitions.requires_grad_(False)
        self.crf.start_transitions.requires_grad_(False)
        self.crf.end_transitions.requires_grad_(False)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs[0])
        emissions = self.classifier(sequence_output)
        mask = attention_mask.type(torch.uint8)

        if labels is not None:
            safe_labels = torch.where(labels >= 0, labels, torch.zeros_like(labels))
            loss = -self.crf(emissions, tags=safe_labels, mask=mask, reduction="mean")
            return loss
        else:
            tags = self.crf.decode(emissions, mask=mask)
            return tags, emissions


# ==========================================
# Address Parser
# ==========================================
class HKAddressParserBERT:
    def __init__(self, model_path, conf_threshold=0.50, max_len=128):
        """
        model_path : folder that contains
                     - config.json
                     - tokenizer files (tokenizer_config.json, sentencepiece.bpe.model, etc.)
                     - pytorch_model.bin
        conf_threshold : if split_conf < this value, emit a low-confidence warning
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.max_len = max_len
        self.device = self._get_emptiest_gpu_safely()
        print(f"DEBUG: Using Device → {self.device}")

        # 1. Load Tokenizer & Config
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.config = AutoConfig.from_pretrained(model_path)
        except Exception as e:
            raise FileNotFoundError(f"❌ Failed to load tokenizer/config from {model_path}. Error: {e}")

        # 2. Extract labels from Config
        # Ensure id2label uses integer keys (JSON loading converts them to strings)
        self.idx2tag = {int(k): v for k, v in self.config.id2label.items()}
        self.label_list = [self.idx2tag[i] for i in range(len(self.idx2tag))]

        # 3. Init Model Architecture
        self.model = BertCRFForTokenClassification(self.config, self.label_list)

        # 4. Load Weights
        weights_path = os.path.join(model_path, "pytorch_model.bin")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"❌ No weights found in {model_path}")

        self._load_weights_with_progress(weights_path)

        self.model.to(self.device)
        self.model.eval()

    def _load_weights_with_progress(self, weights_path):
        print(f"📦 Loading weights from {weights_path} ...")
        state_dict = torch.load(weights_path, map_location="cpu", mmap=True)

        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

        model_keys = set(self.model.state_dict().keys())
        loaded = 0
        missing = []

        with tqdm(total=len(model_keys), desc="Loading weights", unit="tensor") as pbar:
            own = self.model.state_dict()
            for name in list(own.keys()):
                if name in state_dict and own[name].shape == state_dict[name].shape:
                    own[name].copy_(state_dict[name])
                    loaded += 1
                else:
                    missing.append(name)
                pbar.update(1)
                pbar.set_postfix(loaded=loaded)

        unexpected = [k for k in state_dict if k not in model_keys]
        print(f"✅ Loaded {loaded}/{len(model_keys)} tensors | "
              f"missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print("   Missing keys:", missing[:8], "..." if len(missing) > 8 else "")

    @staticmethod
    def _get_emptiest_gpu_safely():
        if not torch.cuda.is_available():
            return torch.device("cpu")
        try:
            result = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
                 "--format=csv,nounits,noheader"], encoding="utf-8")
            best_id, max_free, fallback_id = -1, 0, 0
            for line in result.strip().split("\n"):
                parts = line.split(", ")
                gid, free, util = int(parts[0]), int(parts[1]), int(parts[2])
                if free > max_free and util < 30:
                    max_free, best_id = free, gid
                if free > 0:
                    fallback_id = gid
            return torch.device(f"cuda:{best_id if best_id != -1 else fallback_id}")
        except Exception:
            return torch.device("cuda:0")

    @staticmethod
    def _extract_3d_components(parsed_entities):
        components = defaultdict(list)
        confs = defaultdict(list)
        for ent in parsed_entities:
            tag = ent["entity_group"]
            if tag == "O":
                continue
            components[tag].append(ent["word"])
            confs[tag].append(ent["conf"])

        formatted = {k: "".join(v) for k, v in components.items()}
        conf_out = {k: sum(v) / len(v) for k, v in confs.items()}
        return formatted, conf_out

    @staticmethod
    def _map_extracted_to_standard_keys(extracted_data, conf_data=None):
        mapping = {
            "UNIT": "flat", "FLOOR": "floor", "BLOCK": "block", "PHASE": "phase",
            "BUILDING_NAME": "building_name", "ESTATE_NAME": "estate_name",
            "VILLAGE_NAME": "village_name", "BUILDING_NUMBER": "building_number",
            "STREET_NAME": "street_name", "SUB_DISTRICT": "sub_district",
            "DISTRICT": "district", "REGION": "region"
        }
        mapped = {mapping.get(k, k.lower()): v for k, v in extracted_data.items()}
        if conf_data is not None:
            mapped_conf = {mapping.get(k, k.lower()): v for k, v in conf_data.items()}
            return mapped, mapped_conf
        return mapped

    @staticmethod
    def _split_address(extracted_labels, original_input, parsed_entities, conf_mapped):
        line1_keys = set()
        logic_keys_used = set()

        if "flat" in extracted_labels:
            line1_keys.add("flat")
            logic_keys_used.add("flat")
        if "floor" in extracted_labels:
            line1_keys.add("floor")
            logic_keys_used.add("floor")

        has_bldg = "building_name" in extracted_labels
        has_block = "block" in extracted_labels
        has_est = "estate_name" in extracted_labels
        has_phase = "phase" in extracted_labels
        has_vill = "village_name" in extracted_labels
        has_street = "street_name" in extracted_labels
        has_bldg_no = "building_number" in extracted_labels

        if has_block:
            line1_keys.add("block")
            logic_keys_used.add("block")
            if has_bldg and has_est:
                line1_keys.add("building_name")
                logic_keys_used.add("building_name")
            elif has_bldg and not has_est:
                line1_keys.add("building_name")
                logic_keys_used.add("building_name")
        elif has_bldg:
            line1_keys.add("building_name")
            logic_keys_used.add("building_name")
        elif has_est:
            line1_keys.add("estate_name")
            logic_keys_used.add("estate_name")
            if has_phase:
                line1_keys.add("phase")
                logic_keys_used.add("phase")
        elif has_vill:
            line1_keys.add("village_name")
            logic_keys_used.add("village_name")
            if has_bldg_no:
                line1_keys.add("building_number")
                logic_keys_used.add("building_number")
        elif has_street:
            line1_keys.add("street_name")
            logic_keys_used.add("street_name")
            if has_bldg_no:
                line1_keys.add("building_number")
                logic_keys_used.add("building_number")
        elif has_bldg_no:
            line1_keys.add("building_number")
            logic_keys_used.add("building_number")

        is_chinese = any("\u4e00" <= char <= "\u9fff" for char in original_input)

        mapping = {
            "UNIT": "flat", "FLOOR": "floor", "BLOCK": "block", "PHASE": "phase",
            "BUILDING_NAME": "building_name", "ESTATE_NAME": "estate_name",
            "VILLAGE_NAME": "village_name", "BUILDING_NUMBER": "building_number",
            "STREET_NAME": "street_name", "SUB_DISTRICT": "sub_district",
            "DISTRICT": "district", "REGION": "region"
        }

        token_groups = []
        for entity in parsed_entities:
            raw_tag = entity["entity_group"]
            mapped_tag = mapping.get(raw_tag, raw_tag.lower())
            if mapped_tag in line1_keys:
                token_groups.append("micro")
            elif mapped_tag not in ("o", "O"):
                token_groups.append("macro")
            else:
                token_groups.append("O")

        resolved_groups = []
        last_valid = "micro" if not is_chinese else "macro"
        for tg in token_groups:
            if tg != "O":
                last_valid = tg
                resolved_groups.append(tg)
            else:
                resolved_groups.append(last_valid)

        micro_segments, macro_segments = [], []
        curr_micro_seg, curr_micro_tag = [], None
        curr_macro_seg, curr_macro_tag = [], None

        for entity, group in zip(parsed_entities, resolved_groups):
            tag = mapping.get(entity["entity_group"], entity["entity_group"].lower())
            word = entity["word"]

            if group == "micro":
                if tag not in ("o", "O"):
                    if curr_micro_tag != tag:
                        if curr_micro_seg:
                            micro_segments.append((curr_micro_tag, curr_micro_seg))
                        curr_micro_seg = [word]
                        curr_micro_tag = tag
                    else:
                        curr_micro_seg.append(word)
                else:
                    if curr_micro_seg:
                        curr_micro_seg.append(word)
                    else:
                        curr_micro_seg, curr_micro_tag = [word], "o"
            else:
                if tag not in ("o", "O"):
                    if curr_macro_tag != tag:
                        if curr_macro_seg:
                            macro_segments.append((curr_macro_tag, curr_macro_seg))
                        curr_macro_seg = [word]
                        curr_macro_tag = tag
                    else:
                        curr_macro_seg.append(word)
                else:
                    if curr_macro_seg:
                        curr_macro_seg.append(word)
                    else:
                        curr_macro_seg, curr_macro_tag = [word], "o"

        if curr_micro_seg:
            micro_segments.append((curr_micro_tag, curr_micro_seg))
        if curr_macro_seg:
            macro_segments.append((curr_macro_tag, curr_macro_seg))

        micro_string = "".join("".join(words) for _, words in micro_segments)
        macro_string = "".join("".join(words) for _, words in macro_segments)

        def clean_string(s):
            s = re.sub(r"^[,/\\\-;\s，。、；]+", "", s)
            s = re.sub(r"[,/\\\-;\s，。、；]+$", "", s)
            return re.sub(r"\s{2,}", " ", s).strip()

        micro_string = clean_string(micro_string)
        macro_string = clean_string(macro_string)

        split_conf = 1.0
        for k in logic_keys_used:
            split_conf *= conf_mapped.get(k, 1.0)
        if not logic_keys_used:
            for k, c in conf_mapped.items():
                if k not in ("district", "region", "sub_district"):
                    split_conf *= c

        is_reversed = False
        if is_chinese:
            seen_second = False
            for g in resolved_groups:
                if g == "micro":
                    seen_second = True
                elif g == "macro" and seen_second:
                    is_reversed = True
                    break
        else:
            seen_second = False
            for g in resolved_groups:
                if g == "macro":
                    seen_second = True
                elif g == "micro" and seen_second:
                    is_reversed = True
                    break

        line1 = macro_string if is_chinese else micro_string
        line2 = micro_string if is_chinese else macro_string
        return line1, line2, split_conf, list(logic_keys_used), is_reversed

    def parse_batch(self, address_pairs, batch_size=32):
        all_results = []
        for i in range(0, len(address_pairs), batch_size):
            batch_pairs = address_pairs[i:i + batch_size]
            full_addresses = []

            for p1, p2 in batch_pairs:
                parts = [p.strip() for p in (p1, p2) if p and p.strip()]
                if not parts:
                    full_addresses.append("")
                elif len(parts) == 1:
                    full_addresses.append(parts[0])
                else:
                    combined = parts[0] + parts[1]
                    is_chinese = any("\u4e00" <= c <= "\u9fff" for c in combined)
                    if is_chinese:
                        full_addresses.append(" ".join(parts))
                    else:
                        if parts[0].endswith(","):
                            full_addresses.append(f"{parts[0]} {parts[1]}")
                        else:
                            full_addresses.append(f"{parts[0]}, {parts[1]}")

            # Keep track of indices that are not empty strings to process
            valid_indices = [idx for idx, addr in enumerate(full_addresses) if addr]
            valid_addresses = [full_addresses[idx] for idx in valid_indices]

            if valid_addresses:
                # 1. Tokenize using AutoTokenizer
                encoded = self.tokenizer(
                    valid_addresses,
                    padding=True,
                    truncation=True,
                    max_length=self.max_len,
                    return_offsets_mapping=True,
                    return_tensors="pt"
                )

                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded["attention_mask"].to(self.device)
                offsets_batch = encoded["offset_mapping"].cpu().numpy()

                try:
                    with torch.no_grad():
                        pred_ids_batch, emissions_batch = self.model(input_ids, attention_mask)
                        batch_probs = torch.softmax(emissions_batch, dim=-1).cpu()
                except Exception as e:
                    all_results.extend([("", "", "", {}, {}, 0.0, [], f"ERROR: {e}")] * len(batch_pairs))
                    continue

            # Zip everything back together including blanks
            valid_idx_ptr = 0
            for idx, address_str in enumerate(full_addresses):
                if not address_str:
                    all_results.append(("", "", "", {}, {}, 0.0, [], "EMPTY_INPUT"))
                    continue

                prediction_ids = pred_ids_batch[valid_idx_ptr]
                probs = batch_probs[valid_idx_ptr]
                offsets = offsets_batch[valid_idx_ptr]
                valid_idx_ptr += 1

                char_tags = ["O"] * len(address_str)
                char_confs = [0.0] * len(address_str)

                # 2. Map Subword tokens to string characters
                for j, tag_id in enumerate(prediction_ids):
                    start, end = offsets[j]
                    if start == end:
                        continue  # Special tokens like [CLS], [SEP]

                    tag = self.label_list[tag_id]
                    conf = probs[j, tag_id].item()

                    for c in range(start, min(end, len(char_tags))):
                        # Ensure we don't overwrite the first assigned subword token class (usually the most accurate)
                        if char_tags[c] == "O":
                            char_tags[c] = tag
                            char_confs[c] = conf

                # 3. Use BiLSTM regex to reconstruct formatted entities
                parsed_entities = []
                for match in re.finditer(r'([a-zA-Z0-9]+|[\u4e00-\u9fff]|[^\s])(\s*)', address_str):
                    token_str = match.group(1)
                    trailing = match.group(2)
                    start_idx = match.start(1)
                    tag = char_tags[start_idx] if start_idx < len(char_tags) else "O"
                    conf = char_confs[start_idx] if start_idx < len(char_confs) else 0.0

                    entity_group = tag.replace("B-", "").replace("I-", "") if tag != "O" else "O"
                    parsed_entities.append({
                        "entity_group": entity_group,
                        "word": token_str + trailing,
                        "conf": conf,
                        "start": start_idx,
                    })

                # 4. Standard mapping & formatting
                extracted_raw, conf_raw = self._extract_3d_components(parsed_entities)
                extracted_mapped, conf_mapped = self._map_extracted_to_standard_keys(
                    extracted_raw, conf_raw)

                line1, line2, split_conf, used_keys, is_reversed = self._split_address(
                    extracted_mapped, address_str, parsed_entities, conf_mapped)

                # 5. Status generation logic
                warnings = []
                alerts = []
                if is_reversed:
                    alerts.append("reversed order after parsing")
                if split_conf < self.conf_threshold:
                    warnings.append(
                        f"low splitting confidence ({split_conf:.4f} < {self.conf_threshold})"
                    )

                status = ""
                if alerts:
                    if len(alerts) == 1:
                        status += "[ALERT]: " + alerts[0]
                    else:
                        status += "[ALERTS]: " + "; ".join(alerts)
                if warnings:
                    if len(warnings) == 1:
                        status += "[WARNING]: " + warnings[0]
                    else:
                        status += "[WARNINGS]: " + "; ".join(warnings)
                if not status:
                    status = "[SUCCESS]"

                all_results.append((
                    address_str, line1, line2,
                    extracted_mapped, conf_mapped, split_conf, used_keys, status
                ))

        return all_results

    def parse(self, address_part1, address_part2=""):
        return self.parse_batch([(address_part1, address_part2)], batch_size=1)[0]


# ==========================================
# Example usage
# ==========================================
if __name__ == "__main__":
    # Example: point this to a checkpoint folder containing `config.json`,
    # `tokenizer_config.json`, and `pytorch_model.bin`
    MODEL_DIR = "xlm_roberta_large_crfV4"

    print(f"Loading BERT model from {MODEL_DIR}...")
    parser = HKAddressParserBERT(
        model_path=MODEL_DIR,
    )

    test_cases = [
        ("1M 16 lung sum avenue sheung shui north N.T", ""),
        ("27LD, Block 5, Hemera, Lohas Park, Lohas Road 1, TKO, HK", ""),
        ("21A / 5th Fl., Metroplaza Tower A, Nº 223 Hing Fong Road, Kwai Fong, N.T.", ""),
        ("長洲東灣東堤小築彌敦道一百二十三12座H室5樓", ""),
        ("長洲東灣東堤小築彌敦道一百二十三号12座H室5樓", ""),
        ("深水埗白田街123至125白田邨1号楼, ４０４室Part 1", ""),
        ("Flat A, UG, Happy Mansion, 28 Lockhart Road, Wan Chai, Hong Kong", ""),
    ]

    print("\nRunning tests:\n" + "=" * 70)
    results = parser.parse_batch(test_cases, batch_size=4)

    for i, (a1, a2) in enumerate(test_cases):
        addr, l1, l2, tags, confs, overall, used_keys, status = results[i]
        print(f"Original : {addr}")
        print(f"Line 1   : {l1}")
        print(f"Line 2   : {l2}")
        print(f"Tags     : {tags}")
        print(f"Conf     : {{{', '.join(f'{k}: {v:.3f}' for k, v in confs.items())}}}")
        print(f"Logic keys: {used_keys}  |  Split conf: {overall:.4f}")
        print(f"status   : {status}")
        print("-" * 70)