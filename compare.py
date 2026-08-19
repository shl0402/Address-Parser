#!/usr/bin/env python3
"""
HK Address Parser - Head-to-Head Model Comparison + Confidence-Weighted Voting Ensemble
Runs both BERT-CRF and BiLSTM-CNN-CRF on the same dataset.
Supports:
  - TXT : input_address | GT_line1 | GT_line2
  - JSONL : nested tags or plain line strings
Voting ensemble:
  - Compute each model's average split_conf over the dataset
  - Models that are ALWAYS high-confidence get LOWER voting power
    (weight = 1 / avg_conf) so overconfident models are down-weighted
  - Per-field: if models disagree, pick higher (model_weight * field_conf)
  - Report ensemble accuracy in the same metric table

Enhanced logging:
  - Always writes original expected GT Line1 / Line2
  - Writes GT tags (when available)
  - Writes each model's field tags + per-field confidences
  - Writes model Line1 / Line2 + split_conf + used_keys
  - Writes ensemble tags + lines
  - Makes it easy to inspect exactly how each model classified every tag
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import re
import json
import time
import torch
import subprocess
import torch.nn as nn
from collections import defaultdict, Counter
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig, AutoModel
from torchcrf import CRF
torch.backends.cudnn.enabled = False
# ==========================================
# CONFIGURATION
# ==========================================
BERT_MODEL_DIR = "./xlm_roberta_large_checkpointV3"
BILSTM_MODEL_DIR = "./bilstm_crf_modelV5"
COMPARISON_LOG_FILE = "model_comparison_results.log"
TEST_FILE = "data2/address_dataset.jsonl"
# TEST_FILE = "data2/test_cleaned.jsonl"
BATCH_SIZE = 32
MAX_LEN = 128
WORD_EMBED_DIM = 300
CHAR_EMBED_DIM = 100
CHAR_CNN_FILTERS = 100
HIDDEN_DIM = 256 # was 512
NUM_LSTM_LAYERS = 1 # was 2
POS_EMBED_DIM = 16 # NEW
DROPOUT = 0.5
ALL_FIELDS = [
    "flat", "floor", "building_name", "block", "phase",
    "estate_name", "village_name", "building_number",
    "street_name", "sub_district", "district", "region"
]
# ==========================================
# UTILITY FUNCTIONS
# ==========================================
def normalize_for_eval(text):
    if not text:
        return ""
    return re.sub(r'[\s,/\\\-;\.，。、；]+', '', str(text).lower())
def content_covered(gt_text, address_norm):
    gt_norm = normalize_for_eval(gt_text)
    if not gt_norm:
        return True
    gt_cnt = Counter(gt_norm)
    addr_cnt = Counter(address_norm)
    return all(addr_cnt[c] >= cnt for c, cnt in gt_cnt.items())
def tokenize_text(input_text):
    tokens = []
    for match in re.finditer(r'[a-zA-Z]+|[0-9]+|[\u4e00-\u9fff]|[^\s]', input_text):
        tokens.append(match.group())
    return tokens
def prepare_batch(batch_texts, w2i, c2i):
    batch_tokens = [tokenize_text(text) for text in batch_texts]
    max_seq_len = max(len(t) for t in batch_tokens) if batch_tokens else 1
    max_word_len = max([1] + [len(c) for t in batch_tokens for c in t])
    b_words, b_chars, b_masks = [], [], []
    for tokens in batch_tokens:
        seq_len = len(tokens)
        word_ids = [w2i.get(t, w2i.get("<UNK>", 1)) for t in tokens]
        char_ids_list = [[c2i.get(c, c2i.get("<UNK>", 1)) for c in token] for token in tokens]
        b_words.append(word_ids + [0] * (max_seq_len - seq_len))
        b_masks.append([True] * seq_len + [False] * (max_seq_len - seq_len))
        padded_chars = [chars + [0] * (max_word_len - len(chars)) for chars in char_ids_list]
        padded_chars.extend([[0] * max_word_len for _ in range(max_seq_len - seq_len)])
        b_chars.append(padded_chars)
    return (
        torch.tensor(b_words, dtype=torch.long),
        torch.tensor(b_chars, dtype=torch.long),
        torch.tensor(b_masks, dtype=torch.bool),
        batch_tokens
    )
def load_test_file(path):
    data = []
    is_jsonl = path.lower().endswith(".jsonl") or path.lower().endswith(".json")
    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if is_jsonl:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    print(f"⚠️ Skipping malformed JSON line {line_no}")
                    continue
                inp = item.get("input", "").strip()
                out = item.get("output", {})
                gt_l1, gt_l2, gt_tags, has_tags = _parse_output(out)
            else:
                if " | " in line:
                    parts = [p.strip() for p in line.split(" | ")]
                else:
                    parts = [p.strip() for p in line.split("|")]
                if len(parts) < 2:
                    print(f"⚠️ Skipping malformed TXT line {line_no}")
                    continue
                inp = parts[0]
                gt_l1 = parts[1] if len(parts) > 1 else ""
                gt_l2 = parts[2] if len(parts) > 2 else ""
                gt_tags = None
                has_tags = False
            data.append({
                "input": inp,
                "output": {"line1": gt_l1, "line2": gt_l2},
                "gt_tags": gt_tags,
                "has_tags": has_tags
            })
    return data
def _parse_output(out):
    if not isinstance(out, dict):
        return str(out), "", None, False
    l1 = out.get("line1", "")
    l2 = out.get("line2", "")
    if isinstance(l1, dict) or isinstance(l2, dict):
        tags = {}
        if isinstance(l1, dict):
            tags.update({k: v for k, v in l1.items() if v})
        if isinstance(l2, dict):
            tags.update({k: v for k, v in l2.items() if v})
        return "", "", tags, True
    return str(l1).strip(), str(l2).strip(), None, False
# ==========================================
# SHARED SPLIT LOGIC (no reorder)
# ==========================================
def split_address(extracted_labels, original_input, parsed_entities, conf_mapped):
    line1_keys = set()
    logic_keys_used = set()
    if "flat" in extracted_labels:
        line1_keys.add("flat"); logic_keys_used.add("flat")
    if "floor" in extracted_labels:
        line1_keys.add("floor"); logic_keys_used.add("floor")
    has_bldg = "building_name" in extracted_labels
    has_block = "block" in extracted_labels
    has_est = "estate_name" in extracted_labels
    has_phase = "phase" in extracted_labels
    has_vill = "village_name" in extracted_labels
    has_street = "street_name" in extracted_labels
    has_bldg_no = "building_number" in extracted_labels
    # ---- new building / block / estate logic ----
    if has_block:
        # block always goes to micro
        line1_keys.add("block")
        logic_keys_used.add("block")
        if has_bldg and has_est:
            # block + building → micro, estate → macro
            line1_keys.add("building_name")
            logic_keys_used.add("building_name")
            # estate intentionally left out of line1_keys
        elif has_bldg and not has_est:
            # block → micro, building → macro
            # (do NOT add building_name)
            pass
        # else: only block → already handled
    elif has_bldg:
        # no block
        line1_keys.add("building_name")
        logic_keys_used.add("building_name")
        # if estate also exists it stays macro (not added)
    elif has_est:
        # no block, no building → estate becomes micro
        line1_keys.add("estate_name")
        logic_keys_used.add("estate_name")
        if has_phase:
            line1_keys.add("phase")
            logic_keys_used.add("phase")
    # ---------------------------------------------
    elif has_vill:
        line1_keys.add("village_name"); logic_keys_used.add("village_name")
        if has_bldg_no:
            line1_keys.add("building_number"); logic_keys_used.add("building_number")
    elif has_street:
        line1_keys.add("street_name"); logic_keys_used.add("street_name")
        if has_bldg_no:
            line1_keys.add("building_number"); logic_keys_used.add("building_number")
    elif has_bldg_no:
        line1_keys.add("building_number"); logic_keys_used.add("building_number")
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
    line1 = macro_string if is_chinese else micro_string
    line2 = micro_string if is_chinese else macro_string
    return line1, line2, split_conf, list(logic_keys_used)
# ==========================================
# BERT
# ==========================================
class BertCRFForTokenClassification(torch.nn.Module):
    def __init__(self, config, model_name_or_path, label_list):
        super().__init__()
        self.config = config
        self.label_list = label_list
        self.bert = AutoModel.from_pretrained(model_name_or_path, config=config, ignore_mismatched_sizes=True)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)
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
            return -self.crf(emissions, tags=safe_labels, mask=mask, reduction='mean')
        else:
            tags = self.crf.decode(emissions, mask=mask)
            return tags, emissions
class HKAddressParserBERT:
    def __init__(self, model_path, device):
        self.device = device
        self.config = AutoConfig.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.label_list = [
            self.config.id2label[k] if isinstance(k, int) else self.config.id2label[str(k)]
            for k in sorted(int(k) for k in self.config.id2label.keys())
        ]
        self.model = BertCRFForTokenClassification(self.config, model_path, self.label_list)
        weights_path = os.path.join(model_path, "pytorch_model.bin")
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
    def _extract_components(self, parsed_entities):
        components, confs = defaultdict(list), defaultdict(list)
        for entity in parsed_entities:
            tag = entity["entity_group"]
            if tag == "O":
                continue
            components[tag].append(entity["word"])
            confs[tag].append(entity["conf"])
        formatted = {tag: "".join(words) for tag, words in components.items()}
        conf_out = {tag: sum(cs) / len(cs) if cs else 0.0 for tag, cs in confs.items()}
        return formatted, conf_out
    def _map_keys(self, extracted, confs=None):
        mapping = {
            "UNIT": "flat", "FLOOR": "floor", "BLOCK": "block", "PHASE": "phase",
            "BUILDING_NAME": "building_name", "ESTATE_NAME": "estate_name",
            "VILLAGE_NAME": "village_name", "BUILDING_NUMBER": "building_number",
            "STREET_NAME": "street_name", "SUB_DISTRICT": "sub_district",
            "DISTRICT": "district", "REGION": "region"
        }
        mapped = {mapping.get(k, k.lower()): v for k, v in extracted.items()}
        if confs is not None:
            mapped_conf = {mapping.get(k, k.lower()): v for k, v in confs.items()}
            return mapped, mapped_conf
        return mapped
    def parse_batch(self, full_addresses, batch_size=32):
        all_results = []
        for i in tqdm(range(0, len(full_addresses), batch_size), desc="BERT Inference"):
            batch = full_addresses[i:i + batch_size]
            encoded = self.tokenizer(
                batch, padding=True, truncation=True, max_length=MAX_LEN,
                return_offsets_mapping=True, return_tensors="pt"
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            offsets_batch = encoded["offset_mapping"].cpu().numpy()
            with torch.no_grad():
                batch_predictions, batch_emissions = self.model(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                batch_probs = torch.softmax(batch_emissions, dim=-1).cpu()
            for idx, address_str in enumerate(batch):
                prediction_ids = batch_predictions[idx]
                offsets = offsets_batch[idx]
                item_probs = batch_probs[idx]
                char_tags = ["O"] * len(address_str)
                char_confs = [0.0] * len(address_str)
                for j, tag_id in enumerate(prediction_ids):
                    start, end = offsets[j]
                    if start == end:
                        continue
                    tag = self.label_list[tag_id]
                    conf = item_probs[j, tag_id].item()
                    for c in range(start, end):
                        if char_tags[c] == "O":
                            char_tags[c] = tag
                            char_confs[c] = conf
                parsed_entities = []
                for match in re.finditer(r"([a-zA-Z]+|[0-9]+|[\u4e00-\u9fff]|[^\s])(\s*)", address_str):
                    token_str = match.group(1)
                    trailing = match.group(2)
                    start_idx = match.start(1)
                    tag = char_tags[start_idx] if start_idx < len(char_tags) else "O"
                    conf = char_confs[start_idx] if start_idx < len(char_confs) else 0.0
                    entity_group = tag.replace("B-", "").replace("I-", "") if tag != "O" else "O"
                    parsed_entities.append({
                        "entity_group": entity_group,
                        "word": token_str + trailing,
                        "conf": conf
                    })
                extracted_raw, conf_raw = self._extract_components(parsed_entities)
                extracted, confs = self._map_keys(extracted_raw, conf_raw)
                line1, line2, split_conf, used_keys = split_address(
                    extracted, address_str, parsed_entities, confs
                )
                all_results.append({
                    "tags": extracted, "confs": confs,
                    "line1": line1, "line2": line2,
                    "split_conf": split_conf, "used_keys": used_keys,
                    # Keep raw token-level classification for detailed logging
                    "token_entities": parsed_entities
                })
        return all_results
# ==========================================
# BiLSTM
# ==========================================
class BiLSTM_CNN_CRF(nn.Module):
    def __init__(self, vocab_size, char_vocab_size, num_tags, word_dim, char_dim, cnn_filters,
                 hidden_dim, num_layers=1, pos_dim=16, dropout=0.5, max_seq_len=512):
        super().__init__()
        self.word_embed = nn.Embedding(vocab_size, word_dim, padding_idx=0)
        self.char_embed = nn.Embedding(char_vocab_size, char_dim, padding_idx=0)
        self.char_cnn = nn.Conv1d(in_channels=char_dim, out_channels=cnn_filters, kernel_size=3, padding=1)
        # NEW: positional embedding
        self.pos_embed = nn.Embedding(max_seq_len, pos_dim)
        lstm_input_dim = word_dim + cnn_filters + pos_dim
        self.lstm = nn.LSTM(lstm_input_dim, hidden_dim // 2, num_layers=num_layers,
                            bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.hidden2tag = nn.Linear(hidden_dim, num_tags)
        self.crf = CRF(num_tags, batch_first=True)
    def forward(self, word_ids, char_ids, mask, labels=None):
        batch_size, seq_len = word_ids.shape
        max_word_len = char_ids.shape[2]
        w_emb = self.word_embed(word_ids)
        char_ids_flat = char_ids.view(-1, max_word_len)
        c_emb = self.char_embed(char_ids_flat).permute(0, 2, 1)
        c_cnn_out, _ = torch.max(self.char_cnn(c_emb), dim=2)
        c_features = c_cnn_out.view(batch_size, seq_len, -1)
        # NEW: positional embeddings
        positions = torch.arange(seq_len, device=word_ids.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embed(positions)
        lstm_in = self.dropout(torch.cat([w_emb, c_features, pos_emb], dim=2))
        lstm_out, _ = self.lstm(lstm_in)
        emissions = self.hidden2tag(lstm_out)
        if labels is not None:
            return -self.crf(emissions, tags=labels, mask=mask, reduction='mean')
        return self.crf.decode(emissions, mask=mask), emissions
class HKAddressParserBiLSTM:
    def __init__(self, model_path, device):
        self.device = device
        vocab_path = os.path.join(model_path, "vocabs.json")
        weights_path = os.path.join(model_path, "pytorch_model.bin")
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocabs = json.load(f)
        self.w2i = vocabs["w2i"]
        self.c2i = vocabs["c2i"]
        self.t2i = vocabs["t2i"]
        self.idx2tag = {int(v): k for k, v in self.t2i.items()}
        # === Updated instantiation ===
        self.model = BiLSTM_CNN_CRF(
            vocab_size=len(self.w2i),
            char_vocab_size=len(self.c2i),
            num_tags=len(self.t2i),
            word_dim=WORD_EMBED_DIM,
            char_dim=CHAR_EMBED_DIM,
            cnn_filters=CHAR_CNN_FILTERS,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LSTM_LAYERS,
            pos_dim=POS_EMBED_DIM, # NEW
            dropout=DROPOUT
        )
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
    def _extract_components(self, parsed_entities):
        components, confs = defaultdict(list), defaultdict(list)
        for entity in parsed_entities:
            tag = entity["entity_group"]
            if tag == "O":
                continue
            components[tag].append(entity["word"])
            confs[tag].append(entity["conf"])
        formatted = {tag: "".join(words) for tag, words in components.items()}
        conf_out = {tag: sum(cs) / len(cs) if cs else 0.0 for tag, cs in confs.items()}
        return formatted, conf_out
    def _map_keys(self, extracted, confs=None):
        mapping = {
            "UNIT": "flat", "FLOOR": "floor", "BLOCK": "block", "PHASE": "phase",
            "BUILDING_NAME": "building_name", "ESTATE_NAME": "estate_name",
            "VILLAGE_NAME": "village_name", "BUILDING_NUMBER": "building_number",
            "STREET_NAME": "street_name", "SUB_DISTRICT": "sub_district",
            "DISTRICT": "district", "REGION": "region"
        }
        mapped = {mapping.get(k, k.lower()): v for k, v in extracted.items()}
        if confs is not None:
            mapped_conf = {mapping.get(k, k.lower()): v for k, v in confs.items()}
            return mapped, mapped_conf
        return mapped
    def parse_batch(self, full_addresses, batch_size=32):
        all_results = []
        for i in tqdm(range(0, len(full_addresses), batch_size), desc="BiLSTM Inference"):
            batch = full_addresses[i:i + batch_size]
            word_ids, char_ids, masks, batch_tokens = prepare_batch(batch, self.w2i, self.c2i)
            word_ids = word_ids.to(self.device)
            char_ids = char_ids.to(self.device)
            masks = masks.to(self.device)
            with torch.no_grad():
                prediction_ids_batch, emissions_batch = self.model(word_ids, char_ids, masks)
                batch_probs = torch.softmax(emissions_batch, dim=-1).cpu()
            for idx, address_str in enumerate(batch):
                prediction_ids = prediction_ids_batch[idx]
                item_probs = batch_probs[idx]
                tokens = batch_tokens[idx]
                char_tags = ["O"] * len(address_str)
                char_confs = [0.0] * len(address_str)
                start_indices = [
                    m.start() for m in re.finditer(
                        r'[a-zA-Z]+|[0-9]+|[\u4e00-\u9fff]|[^\s]', address_str
                    )
                ]
                for j, tag_id in enumerate(prediction_ids):
                    if j >= len(start_indices):
                        break
                    start_idx = start_indices[j]
                    tag = self.idx2tag[tag_id]
                    conf = item_probs[j, tag_id].item()
                    for c in range(start_idx, start_idx + len(tokens[j])):
                        if char_tags[c] == "O":
                            char_tags[c] = tag
                            char_confs[c] = conf
                parsed_entities = []
                for match in re.finditer(r"([a-zA-Z]+|[0-9]+|[\u4e00-\u9fff]|[^\s])(\s*)", address_str):
                    token_str = match.group(1)
                    trailing = match.group(2)
                    start_idx = match.start(1)
                    tag = char_tags[start_idx]
                    conf = char_confs[start_idx]
                    entity_group = tag.replace("B-", "").replace("I-", "") if tag != "O" else "O"
                    parsed_entities.append({
                        "entity_group": entity_group,
                        "word": token_str + trailing,
                        "conf": conf
                    })
                extracted_raw, conf_raw = self._extract_components(parsed_entities)
                extracted, confs = self._map_keys(extracted_raw, conf_raw)
                line1, line2, split_conf, used_keys = split_address(
                    extracted, address_str, parsed_entities, confs
                )
                all_results.append({
                    "tags": extracted, "confs": confs,
                    "line1": line1, "line2": line2,
                    "split_conf": split_conf, "used_keys": used_keys,
                    # Keep raw token-level classification for detailed logging
                    "token_entities": parsed_entities
                })
        return all_results
def get_emptiest_gpu_safely():
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
             "--format=csv,nounits,noheader"],
            encoding="utf-8"
        )
        best_id, max_free_mb = 0, 0
        for line in result.strip().split("\n"):
            parts = line.split(", ")
            gpu_id, free_memory, gpu_util = int(parts[0]), int(parts[1]), int(parts[2])
            if gpu_util < 30 and free_memory > max_free_mb:
                max_free_mb, best_id = free_memory, gpu_id
        return torch.device(f"cuda:{best_id}")
    except Exception:
        return torch.device("cuda:0")
# ==========================================
# EVALUATION HELPERS
# ==========================================
def evaluate_lines(pred_line1, pred_line2, gt_line1, gt_line2):
    l1 = normalize_for_eval(pred_line1) == normalize_for_eval(gt_line1)
    l2 = normalize_for_eval(pred_line2) == normalize_for_eval(gt_line2)
    return l1, l2, l1 and l2
def evaluate_fields(pred_tags, gt_tags, used_keys=None):
    if not gt_tags:
        return None
    pred_ff = normalize_for_eval(pred_tags.get("floor", "") + pred_tags.get("flat", ""))
    gt_ff = normalize_for_eval(gt_tags.get("floor", "") + gt_tags.get("flat", ""))
    p_bldg = normalize_for_eval(pred_tags.get("building_name", ""))
    p_est = normalize_for_eval(pred_tags.get("estate_name", ""))
    g_bldg = normalize_for_eval(gt_tags.get("building_name", ""))
    g_est = normalize_for_eval(gt_tags.get("estate_name", ""))
    field_correct = {}
    for field in ALL_FIELDS:
        p_norm = normalize_for_eval(pred_tags.get(field, ""))
        g_norm = normalize_for_eval(gt_tags.get(field, ""))
        if p_norm == g_norm:
            field_correct[field] = True
        elif field in ("floor", "flat") and pred_ff == gt_ff and pred_ff != "":
            field_correct[field] = True
        elif field in ("building_name", "estate_name") and (p_bldg == g_est and p_est == g_bldg):
            field_correct[field] = True
        else:
            field_correct[field] = False
    line1_fields = ["flat", "floor", "block", "phase", "building_name", "estate_name"]
    line2_fields = ["village_name", "building_number", "street_name", "sub_district", "district", "region"]
    is_l1 = all(field_correct[f] for f in line1_fields)
    is_l2 = all(field_correct[f] for f in line2_fields)
    if used_keys:
        is_logic = all(field_correct.get(k, False) for k in used_keys)
    else:
        is_logic = True
    return {
        "field_correct": field_correct,
        "l1_fields_ok": is_l1,
        "l2_fields_ok": is_l2,
        "full_fields_ok": is_l1 and is_l2,
        "logic_ok": is_logic
    }
# ==========================================
# VOTING ENSEMBLE
# ==========================================
def compute_model_weights(bert_results, bilstm_results, valid_indices):
    """Higher avg conf -> lower voting weight (overconfident models down-weighted)."""
    bert_confs = [bert_results[i]["split_conf"] for i in valid_indices]
    bilstm_confs = [bilstm_results[i]["split_conf"] for i in valid_indices]
    avg_bert = sum(bert_confs) / len(bert_confs) if bert_confs else 1.0
    avg_bilstm = sum(bilstm_confs) / len(bilstm_confs) if bilstm_confs else 1.0
    w_bert = 1.0 / (avg_bert + 1e-6)
    w_bilstm = 1.0 / (avg_bilstm + 1e-6)
    total = w_bert + w_bilstm
    return w_bert / total, w_bilstm / total, avg_bert, avg_bilstm
def ensemble_vote_tags(bert_res, bilstm_res, w_bert, w_bilstm):
    """Field-wise: score = model_weight * field_conf. Higher wins on disagreement."""
    ens_tags, ens_confs = {}, {}
    all_keys = set(bert_res["tags"].keys()) | set(bilstm_res["tags"].keys())
    for field in all_keys:
        b_val = bert_res["tags"].get(field, "")
        l_val = bilstm_res["tags"].get(field, "")
        b_conf = bert_res["confs"].get(field, 0.0)
        l_conf = bilstm_res["confs"].get(field, 0.0)
        if normalize_for_eval(b_val) == normalize_for_eval(l_val):
            ens_tags[field] = b_val if b_val else l_val
            ens_confs[field] = max(b_conf, l_conf)
        else:
            if w_bert * b_conf >= w_bilstm * l_conf:
                ens_tags[field] = b_val
                ens_confs[field] = b_conf
            else:
                ens_tags[field] = l_val
                ens_confs[field] = l_conf
    return ens_tags, ens_confs
def ensemble_vote_lines(bert_res, bilstm_res, w_bert, w_bilstm):
    """Pick whole prediction from model with higher weight * split_conf."""
    if w_bert * bert_res["split_conf"] >= w_bilstm * bilstm_res["split_conf"]:
        return bert_res["line1"], bert_res["line2"], bert_res["used_keys"]
    return bilstm_res["line1"], bilstm_res["line2"], bilstm_res["used_keys"]
# ==========================================
# DETAILED LOGGING HELPERS
# ==========================================
def format_tags_with_conf(tags, confs):
    """Pretty print field tags together with their confidence scores."""
    if not tags:
        return "{}"
    parts = []
    for k in sorted(tags.keys()):
        conf = confs.get(k, 0.0) if confs else 0.0
        parts.append(f"{k}='{tags[k]}'({conf:.3f})")
    return "{" + ", ".join(parts) + "}"

def format_token_entities(token_entities):
    """Show how the model classified every token (word + entity_group + conf)."""
    if not token_entities:
        return "[]"
    parts = []
    for ent in token_entities:
        parts.append(f"{ent['word'].strip()}→{ent['entity_group']}({ent['conf']:.3f})")
    return " | ".join(parts)

def _decide_line1_keys(extracted_labels):
    """Shared logic to decide which fields belong to the 'micro' / line1 side."""
    line1_keys = set()
    if "flat" in extracted_labels:
        line1_keys.add("flat")
    if "floor" in extracted_labels:
        line1_keys.add("floor")
    has_bldg = "building_name" in extracted_labels
    has_block = "block" in extracted_labels
    has_est = "estate_name" in extracted_labels
    has_phase = "phase" in extracted_labels
    has_vill = "village_name" in extracted_labels
    has_street = "street_name" in extracted_labels
    has_bldg_no = "building_number" in extracted_labels
    if has_block:
        line1_keys.add("block")
        if has_bldg and has_est:
            line1_keys.add("building_name")
        # elif has_bldg and not has_est: building stays macro
    elif has_bldg:
        line1_keys.add("building_name")
    elif has_est:
        line1_keys.add("estate_name")
        if has_phase:
            line1_keys.add("phase")
    elif has_vill:
        line1_keys.add("village_name")
        if has_bldg_no:
            line1_keys.add("building_number")
    elif has_street:
        line1_keys.add("street_name")
        if has_bldg_no:
            line1_keys.add("building_number")
    elif has_bldg_no:
        line1_keys.add("building_number")
    return line1_keys

def reconstruct_gt_lines(gt_tags, original_input):
    """Approximate Line1 / Line2 from field tags when the dataset only provides tags.
    Uses the same micro/macro decision as split_address and concatenates values
    in the canonical ALL_FIELDS order (good enough for human inspection).
    """
    if not gt_tags:
        return "", ""
    extracted = {k for k, v in gt_tags.items() if v}
    line1_keys = _decide_line1_keys(extracted)
    is_chinese = any("\u4e00" <= char <= "\u9fff" for char in original_input)
    micro_parts, macro_parts = [], []
    for f in ALL_FIELDS:
        val = gt_tags.get(f, "")
        if not val:
            continue
        if f in line1_keys:
            micro_parts.append(str(val))
        else:
            macro_parts.append(str(val))
    micro = "".join(micro_parts)
    macro = "".join(macro_parts)
    # same assignment as split_address
    line1 = macro if is_chinese else micro
    line2 = micro if is_chinese else macro
    return line1, line2

def _emoji(ok):
    return "✅" if ok else "❌"

def _pad(s, width):
    s = str(s) if s is not None else ""
    return s[:width].ljust(width)

def write_sample_detail(log, idx, address, gt_l1, gt_l2, gt_tags, has_tags,
                        bert_res, bilstm_res, ens_tags, ens_confs,
                        ens_l1, ens_l2, ens_used,
                        b_full, l_full, e_full,
                        b_l1, b_l2, l_l1, l_l2, e_l1, e_l2,
                        b_logic, l_logic, e_logic):
    """Write a clean, table-oriented block for one sample.
    Shows:
      - Input + Expected (reconstructed lines when original L1/L2 empty)
      - Field-level table with values + confidences + correctness emojis
      - Line1 / Line2 table with match status
    """
    # Reconstruct GT lines if the dataset only gave tags (common cause of empty L1/L2)
    if has_tags and gt_tags and (not gt_l1 and not gt_l2):
        gt_l1, gt_l2 = reconstruct_gt_lines(gt_tags, address)
        gt_lines_note = "  (reconstructed from tags)"
    else:
        gt_lines_note = ""

    log.write("=" * 100 + "\n")
    log.write(f"SAMPLE {idx}\n")
    log.write("=" * 100 + "\n")
    log.write(f"Input     : {address}\n")
    log.write(f"Expected L1: {gt_l1}{gt_lines_note}\n")
    log.write(f"Expected L2: {gt_l2}\n")
    if has_tags and gt_tags:
        log.write(f"Expected tags: {gt_tags}\n")
    log.write("\n")

    # ------------------------------------------------------------------
    # 1) FIELD TAG TABLE (only when ground-truth tags exist)
    # ------------------------------------------------------------------
    if has_tags and gt_tags:
        # Re-evaluate per-field correctness so we can show emojis
        b_eval = evaluate_fields(bert_res["tags"], gt_tags, used_keys=bert_res.get("used_keys"))
        l_eval = evaluate_fields(bilstm_res["tags"], gt_tags, used_keys=bilstm_res.get("used_keys"))
        e_eval = evaluate_fields(ens_tags, gt_tags)

        b_ok = b_eval["field_correct"] if b_eval else {}
        l_ok = l_eval["field_correct"] if l_eval else {}
        e_ok = e_eval["field_correct"] if e_eval else {}

        # Collect every field that appears in GT or any prediction
        all_fields = sorted(
            set(gt_tags.keys()) |
            set(bert_res["tags"].keys()) |
            set(bilstm_res["tags"].keys()) |
            set(ens_tags.keys())
        )
        # Prefer canonical order
        ordered = [f for f in ALL_FIELDS if f in all_fields] + \
                  [f for f in all_fields if f not in ALL_FIELDS]

        log.write("┌─ FIELD TAGS + CONFIDENCE + CORRECTNESS ─────────────────────────────────────────────┐\n")
        log.write("│ {:<16} │ {:<18} │ {:<22} │ {:<22} │ {:<22} │\n".format(
            "Field", "GT", "BERT (conf)", "BiLSTM (conf)", "ENSEMBLE (conf)"))
        log.write("├─{:-<16}─┼─{:-<18}─┼─{:-<22}─┼─{:-<22}─┼─{:-<22}─┤\n".format("", "", "", "", ""))

        for f in ordered:
            gt_val = gt_tags.get(f, "") or ""
            b_val = bert_res["tags"].get(f, "") or ""
            l_val = bilstm_res["tags"].get(f, "") or ""
            e_val = ens_tags.get(f, "") or ""
            b_conf = bert_res["confs"].get(f, 0.0)
            l_conf = bilstm_res["confs"].get(f, 0.0)
            e_conf = ens_confs.get(f, 0.0)

            # emoji only meaningful when GT has the field (or is empty and pred empty)
            b_emoji = _emoji(b_ok.get(f, False)) if f in b_ok else "  "
            l_emoji = _emoji(l_ok.get(f, False)) if f in l_ok else "  "
            e_emoji = _emoji(e_ok.get(f, False)) if f in e_ok else "  "

            # Compact cell: value (conf) emoji
            b_cell = f"{b_val}({b_conf:.2f}){b_emoji}" if b_val or b_conf > 0 else ""
            l_cell = f"{l_val}({l_conf:.2f}){l_emoji}" if l_val or l_conf > 0 else ""
            e_cell = f"{e_val}({e_conf:.2f}){e_emoji}" if e_val or e_conf > 0 else ""

            log.write("│ {:<16} │ {:<18} │ {:<22} │ {:<22} │ {:<22} │\n".format(
                f[:16],
                (gt_val[:16] + ("…" if len(gt_val) > 16 else "")),
                b_cell[:22],
                l_cell[:22],
                e_cell[:22]
            ))
        log.write("└─{:-<16}─┴─{:-<18}─┴─{:-<22}─┴─{:-<22}─┴─{:-<22}─┘\n".format("", "", "", "", ""))
        log.write("\n")

    # ------------------------------------------------------------------
    # 2) LINE1 / LINE2 TABLE (always)
    # ------------------------------------------------------------------
    log.write("┌─ MERGED LINE1 / LINE2 + MATCH STATUS ───────────────────────────────────────────────┐\n")
    log.write("│ {:<10} │ {:<40} │ {:<40} │ {:^6} │ {:^6} │ {:^6} │\n".format(
        "Model", "Line 1", "Line 2", "L1", "L2", "Full"))
    log.write("├─{:-<10}─┼─{:-<40}─┼─{:-<40}─┼─{:-<6}─┼─{:-<6}─┼─{:-<6}─┤\n".format(
        "", "", "", "", "", ""))

    def _short(s, n=38):
        s = str(s) if s else ""
        return (s[:n] + "…") if len(s) > n else s

    # Expected row
    log.write("│ {:<10} │ {:<40} │ {:<40} │ {:^6} │ {:^6} │ {:^6} │\n".format(
        "EXPECTED", _short(gt_l1), _short(gt_l2), "—", "—", "—"))
    # BERT
    log.write("│ {:<10} │ {:<40} │ {:<40} │ {:^6} │ {:^6} │ {:^6} │\n".format(
        "BERT", _short(bert_res["line1"]), _short(bert_res["line2"]),
        _emoji(b_l1), _emoji(b_l2), _emoji(b_full)))
    # BiLSTM
    log.write("│ {:<10} │ {:<40} │ {:<40} │ {:^6} │ {:^6} │ {:^6} │\n".format(
        "BiLSTM", _short(bilstm_res["line1"]), _short(bilstm_res["line2"]),
        _emoji(l_l1), _emoji(l_l2), _emoji(l_full)))
    # Ensemble
    log.write("│ {:<10} │ {:<40} │ {:<40} │ {:^6} │ {:^6} │ {:^6} │\n".format(
        "ENSEMBLE", _short(ens_l1), _short(ens_l2),
        _emoji(e_l1), _emoji(e_l2), _emoji(e_full)))
    log.write("└─{:-<10}─┴─{:-<40}─┴─{:-<40}─┴─{:-<6}─┴─{:-<6}─┴─{:-<6}─┘\n".format(
        "", "", "", "", "", ""))

    # Extra meta
    log.write(f"BERT   split_conf={bert_res['split_conf']:.4f}  used_keys={bert_res.get('used_keys', [])}  logic={_emoji(b_logic)}\n")
    log.write(f"BiLSTM split_conf={bilstm_res['split_conf']:.4f}  used_keys={bilstm_res.get('used_keys', [])}  logic={_emoji(l_logic)}\n")
    log.write(f"ENS    used_keys={ens_used}  logic={_emoji(e_logic)}\n")

    # Optional token-level (compact, only if useful)
    if bert_res.get("token_entities"):
        log.write(f"BERT tokens  : {format_token_entities(bert_res['token_entities'])}\n")
    if bilstm_res.get("token_entities"):
        log.write(f"BiLSTM tokens: {format_token_entities(bilstm_res['token_entities'])}\n")
    log.write("\n")
# ==========================================
# MAIN
# ==========================================
def main():
    if not os.path.exists(BERT_MODEL_DIR) or not os.path.exists(BILSTM_MODEL_DIR):
        print("❌ Error: One or both model directories not found.")
        print(f" BERT_MODEL_DIR = {BERT_MODEL_DIR}")
        print(f" BILSTM_MODEL_DIR = {BILSTM_MODEL_DIR}")
        return
    if not os.path.exists(TEST_FILE):
        print(f"❌ Error: Test file not found: {TEST_FILE}")
        return
    device = get_emptiest_gpu_safely()
    print(f"DEBUG: Using Device -> {device}")
    print("\n📦 Loading BERT Model...")
    parser_bert = HKAddressParserBERT(model_path=BERT_MODEL_DIR, device=device)
    print("📦 Loading BiLSTM Model...")
    parser_bilstm = HKAddressParserBiLSTM(model_path=BILSTM_MODEL_DIR, device=device)
    print(f"\n🚀 Loading dataset from {TEST_FILE}...")
    test_data = load_test_file(TEST_FILE)
    print(f" Loaded {len(test_data)} samples.")
    has_any_tags = any(item["has_tags"] for item in test_data)
    if has_any_tags:
        print(" Detected individual field tags → field-level accuracy + voting.")
    else:
        print(" No individual field tags → line-string accuracy + voting.")
    inputs = [item["input"] for item in test_data]
    print("\n⚡ Running inference for both models...")
    t0 = time.perf_counter()
    bert_results = parser_bert.parse_batch(inputs, batch_size=BATCH_SIZE)
    t_bert = time.perf_counter() - t0
    t0 = time.perf_counter()
    bilstm_results = parser_bilstm.parse_batch(inputs, batch_size=BATCH_SIZE)
    t_bilstm = time.perf_counter() - t0
    # Collect valid indices for weight computation
    valid_indices = []
    for idx in range(len(inputs)):
        address = inputs[idx]
        gt_l1 = test_data[idx]["output"]["line1"]
        gt_l2 = test_data[idx]["output"]["line2"]
        gt_tags = test_data[idx]["gt_tags"]
        has_tags = test_data[idx]["has_tags"]
        norm_addr = normalize_for_eval(address)
        if has_tags and gt_tags:
            if any(normalize_for_eval(v) and normalize_for_eval(v) not in norm_addr
                   for v in gt_tags.values()):
                continue
        else:
            if not content_covered(gt_l1, norm_addr) or not content_covered(gt_l2, norm_addr):
                continue
        valid_indices.append(idx)
    w_bert, w_bilstm, avg_bert, avg_bilstm = compute_model_weights(
        bert_results, bilstm_results, valid_indices
    )
    print(f"\n📊 Model confidence & voting weights:")
    print(f" BERT avg_conf={avg_bert:.4f} → weight={w_bert:.4f}")
    print(f" BiLSTM avg_conf={avg_bilstm:.4f} → weight={w_bilstm:.4f}")
    print(f" (higher avg conf → lower weight)")
    both_correct, both_wrong, bert_only, bilstm_only = [], [], [], []
    excluded = 0
    bert_logic = bert_l1 = bert_l2 = bert_full = 0
    bilstm_logic = bilstm_l1 = bilstm_l2 = bilstm_full = 0
    ens_logic = ens_l1 = ens_l2 = ens_full = 0
    bert_field_ok = Counter()
    bilstm_field_ok = Counter()
    ens_field_ok = Counter()
    field_total = Counter()
    bert_fields_full = bilstm_fields_full = ens_fields_full = 0
    print("\n✍️ Evaluating, voting, and comparing...")
    with open(COMPARISON_LOG_FILE, "w", encoding="utf-8") as log:
        log.write("=" * 100 + "\n")
        log.write("DETAILED SAMPLE-BY-SAMPLE COMPARISON LOG (TABLE FORMAT)\n")
        log.write("Only imperfect samples (not all three full-correct) are listed below.\n")
        log.write("For each sample you will see:\n")
        log.write("  • Input address + Expected L1/L2 (auto-reconstructed from tags when original lines empty)\n")
        log.write("  • FIELD TAGS table: GT | BERT(conf)emoji | BiLSTM(conf)emoji | ENSEMBLE(conf)emoji\n")
        log.write("  • LINE1 / LINE2 table: Expected / BERT / BiLSTM / Ensemble with ✅/❌ match status\n")
        log.write("  • split_conf, used_keys, logic status, and optional token-level tags\n")
        log.write("=" * 100 + "\n\n")
        for idx in range(len(inputs)):
            address = inputs[idx]
            gt_l1 = test_data[idx]["output"]["line1"]
            gt_l2 = test_data[idx]["output"]["line2"]
            gt_tags = test_data[idx]["gt_tags"]
            has_tags = test_data[idx]["has_tags"]
            norm_addr = normalize_for_eval(address)
            if has_tags and gt_tags:
                if any(normalize_for_eval(v) and normalize_for_eval(v) not in norm_addr
                       for v in gt_tags.values()):
                    excluded += 1
                    continue
            else:
                if not content_covered(gt_l1, norm_addr) or not content_covered(gt_l2, norm_addr):
                    excluded += 1
                    continue
            b_full = l_full = e_full = False
            b_l1 = b_l2 = l_l1 = l_l2 = e_l1 = e_l2 = False
            b_logic = l_logic = e_logic = True
            # Always prepare ensemble tags & lines for logging
            ens_tags, ens_confs = ensemble_vote_tags(
                bert_results[idx], bilstm_results[idx], w_bert, w_bilstm
            )
            ens_l1_str, ens_l2_str, ens_used = ensemble_vote_lines(
                bert_results[idx], bilstm_results[idx], w_bert, w_bilstm
            )
            if has_tags and gt_tags:
                b_fields = evaluate_fields(
                    bert_results[idx]["tags"], gt_tags,
                    used_keys=bert_results[idx].get("used_keys")
                )
                l_fields = evaluate_fields(
                    bilstm_results[idx]["tags"], gt_tags,
                    used_keys=bilstm_results[idx].get("used_keys")
                )
                ens_used_for_fields = list(set(
                    bert_results[idx].get("used_keys", []) +
                    bilstm_results[idx].get("used_keys", [])
                ))
                e_fields = evaluate_fields(ens_tags, gt_tags, used_keys=ens_used_for_fields)
                if b_fields and l_fields and e_fields:
                    b_l1, b_l2, b_full = b_fields["l1_fields_ok"], b_fields["l2_fields_ok"], b_fields["full_fields_ok"]
                    b_logic = b_fields["logic_ok"]
                    l_l1, l_l2, l_full = l_fields["l1_fields_ok"], l_fields["l2_fields_ok"], l_fields["full_fields_ok"]
                    l_logic = l_fields["logic_ok"]
                    e_l1, e_l2, e_full = e_fields["l1_fields_ok"], e_fields["l2_fields_ok"], e_fields["full_fields_ok"]
                    e_logic = e_fields["logic_ok"]
                    for f in ALL_FIELDS:
                        if normalize_for_eval(gt_tags.get(f, "")):
                            field_total[f] += 1
                            if b_fields["field_correct"].get(f, False):
                                bert_field_ok[f] += 1
                            if l_fields["field_correct"].get(f, False):
                                bilstm_field_ok[f] += 1
                            if e_fields["field_correct"].get(f, False):
                                ens_field_ok[f] += 1
                    if b_full: bert_fields_full += 1
                    if l_full: bilstm_fields_full += 1
                    if e_full: ens_fields_full += 1
            else:
                b_l1, b_l2, b_full = evaluate_lines(
                    bert_results[idx]["line1"], bert_results[idx]["line2"], gt_l1, gt_l2
                )
                l_l1, l_l2, l_full = evaluate_lines(
                    bilstm_results[idx]["line1"], bilstm_results[idx]["line2"], gt_l1, gt_l2
                )
                e_l1, e_l2, e_full = evaluate_lines(ens_l1_str, ens_l2_str, gt_l1, gt_l2)
                b_logic, l_logic, e_logic = b_full, l_full, e_full
            if b_logic: bert_logic += 1
            if b_l1: bert_l1 += 1
            if b_l2: bert_l2 += 1
            if b_full: bert_full += 1
            if l_logic: bilstm_logic += 1
            if l_l1: bilstm_l1 += 1
            if l_l2: bilstm_l2 += 1
            if l_full: bilstm_full += 1
            if e_logic: ens_logic += 1
            if e_l1: ens_l1 += 1
            if e_l2: ens_l2 += 1
            if e_full: ens_full += 1
            if b_full and l_full:
                both_correct.append(idx)
            elif not b_full and not l_full:
                both_wrong.append(idx)
            elif b_full and not l_full:
                bert_only.append(idx)
            else:
                bilstm_only.append(idx)
            # Only log imperfect samples (keeps log size reasonable)
            if not (b_full and l_full and e_full):
                write_sample_detail(
                    log, idx, address, gt_l1, gt_l2, gt_tags, has_tags,
                    bert_results[idx], bilstm_results[idx],
                    ens_tags, ens_confs, ens_l1_str, ens_l2_str, ens_used,
                    b_full, l_full, e_full,
                    b_l1, b_l2, l_l1, l_l2, e_l1, e_l2,
                    b_logic, l_logic, e_logic
                )
        total_valid = len(inputs) - excluded
        def pct(n, d):
            return (n / d * 100) if d > 0 else 0.0
        metric_note = (
            "FIELD-LEVEL (same definitions as original eval scripts)"
            if has_any_tags else
            "LINE-STRING equality (after split)"
        )
        summary = f"""
=================================================================
🤖 BERT vs BiLSTM + VOTING ENSEMBLE COMPARISON REPORT
=================================================================
Test file : {TEST_FILE}
Total Addresses : {len(inputs)}
Excluded (Corrupted GT) : {excluded}
Total Valid Evaluated : {total_valid}
BERT inference time : {t_bert:.2f}s
BiLSTM inference time : {t_bilstm:.2f}s
Primary metric : {metric_note}
Voting weights (1/avg_conf, normalized):
  BERT avg_conf={avg_bert:.4f} weight={w_bert:.4f}
  BiLSTM avg_conf={avg_bilstm:.4f} weight={w_bilstm:.4f}
-----------------------------------------------------------------
🚀 BERT – ALL PREDICTIONS (Threshold 0%)
-----------------------------------------------------------------
METRIC | ACCURACY (Correct / Total)
-----------------------------------------------------------------
Split Logic Determination Correct | {pct(bert_logic, total_valid):6.2f}% ({bert_logic}/{total_valid})
Line 1 (Micro) Components Correct | {pct(bert_l1, total_valid):6.2f}% ({bert_l1}/{total_valid})
Line 2 (Macro) Components Correct | {pct(bert_l2, total_valid):6.2f}% ({bert_l2}/{total_valid})
Full Address Perfect Match | {pct(bert_full, total_valid):6.2f}% ({bert_full}/{total_valid})
-----------------------------------------------------------------
🚀 BiLSTM – ALL PREDICTIONS (Threshold 0%)
-----------------------------------------------------------------
METRIC | ACCURACY (Correct / Total)
-----------------------------------------------------------------
Split Logic Determination Correct | {pct(bilstm_logic, total_valid):6.2f}% ({bilstm_logic}/{total_valid})
Line 1 (Micro) Components Correct | {pct(bilstm_l1, total_valid):6.2f}% ({bilstm_l1}/{total_valid})
Line 2 (Macro) Components Correct | {pct(bilstm_l2, total_valid):6.2f}% ({bilstm_l2}/{total_valid})
Full Address Perfect Match | {pct(bilstm_full, total_valid):6.2f}% ({bilstm_full}/{total_valid})
-----------------------------------------------------------------
🗳️ ENSEMBLE (confidence-weighted vote) – ALL PREDICTIONS
-----------------------------------------------------------------
METRIC | ACCURACY (Correct / Total)
-----------------------------------------------------------------
Split Logic Determination Correct | {pct(ens_logic, total_valid):6.2f}% ({ens_logic}/{total_valid})
Line 1 (Micro) Components Correct | {pct(ens_l1, total_valid):6.2f}% ({ens_l1}/{total_valid})
Line 2 (Macro) Components Correct | {pct(ens_l2, total_valid):6.2f}% ({ens_l2}/{total_valid})
Full Address Perfect Match | {pct(ens_full, total_valid):6.2f}% ({ens_full}/{total_valid})
-----------------------------------------------------------------
🏆 HEAD-TO-HEAD (Full Address Perfect Match, BERT vs BiLSTM)
-----------------------------------------------------------------
Both Correct : {len(both_correct)}
Both Wrong : {len(both_wrong)}
✅ BERT Right, BiLSTM Wrong : {len(bert_only)}
✅ BiLSTM Right, BERT Wrong : {len(bilstm_only)}
"""
        if has_any_tags and field_total:
            summary += f"""
-----------------------------------------------------------------
📋 FIELD-LEVEL ACCURACY
-----------------------------------------------------------------
{'FIELD':<20} | {'BERT':>16} | {'BiLSTM':>16} | {'ENSEMBLE':>16}
{'-'*20}-+-{'-'*16}-+-{'-'*16}-+-{'-'*16}
"""
            for f in ALL_FIELDS:
                tot = field_total[f]
                if tot == 0:
                    continue
                b_ok, l_ok, e_ok = bert_field_ok[f], bilstm_field_ok[f], ens_field_ok[f]
                summary += (
                    f"{f:<20} | {pct(b_ok, tot):5.1f}% ({b_ok}/{tot})"
                    f" | {pct(l_ok, tot):5.1f}% ({l_ok}/{tot})"
                    f" | {pct(e_ok, tot):5.1f}% ({e_ok}/{tot})\n"
                )
            summary += f"""
Full fields match (all tags correct):
  BERT : {pct(bert_fields_full, total_valid):6.2f}% ({bert_fields_full}/{total_valid})
  BiLSTM : {pct(bilstm_fields_full, total_valid):6.2f}% ({bilstm_fields_full}/{total_valid})
  ENSEMBLE : {pct(ens_fields_full, total_valid):6.2f}% ({ens_fields_full}/{total_valid})
"""
        summary += "=================================================================\n"
        print(summary)
        log.write(summary)
        # Detailed category dumps (also enriched)
        def log_diff(title, indices):
            log.write(f"\n{title}\n" + "-" * 70 + "\n")
            for i in indices:
                address = inputs[i]
                gt_l1 = test_data[i]["output"]["line1"]
                gt_l2 = test_data[i]["output"]["line2"]
                gt_tags = test_data[i]["gt_tags"]
                has_tags = test_data[i]["has_tags"]
                ens_tags, ens_confs = ensemble_vote_tags(
                    bert_results[i], bilstm_results[i], w_bert, w_bilstm
                )
                ens_l1_str, ens_l2_str, ens_used = ensemble_vote_lines(
                    bert_results[i], bilstm_results[i], w_bert, w_bilstm
                )
                # Re-compute status flags for the dump (same logic as main loop)
                if has_tags and gt_tags:
                    b_fields = evaluate_fields(bert_results[i]["tags"], gt_tags,
                                               used_keys=bert_results[i].get("used_keys"))
                    l_fields = evaluate_fields(bilstm_results[i]["tags"], gt_tags,
                                               used_keys=bilstm_results[i].get("used_keys"))
                    e_fields = evaluate_fields(ens_tags, gt_tags)
                    b_full = b_fields["full_fields_ok"] if b_fields else False
                    l_full = l_fields["full_fields_ok"] if l_fields else False
                    e_full = e_fields["full_fields_ok"] if e_fields else False
                    b_l1 = b_fields["l1_fields_ok"] if b_fields else False
                    b_l2 = b_fields["l2_fields_ok"] if b_fields else False
                    l_l1 = l_fields["l1_fields_ok"] if l_fields else False
                    l_l2 = l_fields["l2_fields_ok"] if l_fields else False
                    e_l1 = e_fields["l1_fields_ok"] if e_fields else False
                    e_l2 = e_fields["l2_fields_ok"] if e_fields else False
                    b_logic = b_fields["logic_ok"] if b_fields else True
                    l_logic = l_fields["logic_ok"] if l_fields else True
                    e_logic = e_fields["logic_ok"] if e_fields else True
                else:
                    b_l1, b_l2, b_full = evaluate_lines(
                        bert_results[i]["line1"], bert_results[i]["line2"], gt_l1, gt_l2)
                    l_l1, l_l2, l_full = evaluate_lines(
                        bilstm_results[i]["line1"], bilstm_results[i]["line2"], gt_l1, gt_l2)
                    e_l1, e_l2, e_full = evaluate_lines(ens_l1_str, ens_l2_str, gt_l1, gt_l2)
                    b_logic = b_full
                    l_logic = l_full
                    e_logic = e_full
                write_sample_detail(
                    log, i, address, gt_l1, gt_l2, gt_tags, has_tags,
                    bert_results[i], bilstm_results[i],
                    ens_tags, ens_confs, ens_l1_str, ens_l2_str, ens_used,
                    b_full, l_full, e_full,
                    b_l1, b_l2, l_l1, l_l2, e_l1, e_l2,
                    b_logic, l_logic, e_logic
                )
        log_diff("📌 WHERE BERT WAS CORRECT BUT BiLSTM FAILED", bert_only)
        log_diff("📌 WHERE BiLSTM WAS CORRECT BUT BERT FAILED", bilstm_only)
        log_diff("📌 WHERE BOTH FAILED", both_wrong)
    print(f"\n✅ Done! Detailed comparison saved to {COMPARISON_LOG_FILE}")
    print("   The log now uses clean ASCII tables for every imperfect sample:")
    print("   • Expected L1/L2 (auto-reconstructed from tags when original lines were empty)")
    print("   • Field-tags table: GT | BERT(value+conf+✅/❌) | BiLSTM | ENSEMBLE")
    print("   • Line1/Line2 table with match status emojis for BERT / BiLSTM / Ensemble")
    print("   • split_conf, used_keys, logic status + optional token-level detail")
if __name__ == "__main__":
    main()