#!/usr/bin/env python3
"""
HK Address Parser – Inference for Version 7 (BiLSTM-CNN-CRF + RoPE)
Compatible with models trained by the V7 / V8 grid-search scripts.

Changes:
- Status is no longer always "SUCCESS".
- Emits [ALERT]/ [ALERTS] for reversed order after parsing
  when the micro/macro assignment is not a clean sequential split
  (i.e. the desired group order is violated by interleaving or inversion).
  This correctly flags cases such as:
      天爾天7座50樓h9 啟德協調道62號
      → Line1: 天爾協調道62號
      → Line2: 天7座50樓h9 啟德
- Emits [WARNING]/ [WARNINGS] for low splitting confidence when split_conf < threshold.
- Status construction logic (priority of WARNING over ALERT, singular/plural forms)
  is retained exactly for compatibility with other modules.
"""

import os
import re
import json
import torch
import subprocess
import torch.nn as nn
from collections import defaultdict
from tqdm import tqdm
from torchcrf import CRF

# ==========================================
# DEFAULTS (can be overridden per model)
# ==========================================
WORD_EMBED_DIM = 300
CHAR_EMBED_DIM = 100
CHAR_CNN_FILTERS = 100
ROPE_BASE = 10000.0
MAX_SEQ_LEN_ROPE = 512


# ==========================================
# RoPE helpers (same as training)
# ==========================================
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=512, base=10000.0):
        super().__init__()
        assert dim % 2 == 0, "RoPE dimension must be even"
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_seq_len
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, :, :], persistent=False)
        self.max_seq_len = seq_len

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        cos = self.cos_cached[:, :seq_len, :].to(dtype=x.dtype)
        sin = self.sin_cached[:, :seq_len, :].to(dtype=x.dtype)
        return apply_rotary_pos_emb(x, cos, sin)


# ==========================================
# Model (V7 architecture with RoPE)
# ==========================================
class BiLSTM_CNN_CRF(nn.Module):
    def __init__(self, vocab_size, char_vocab_size, num_tags,
                 word_dim=300, char_dim=100, cnn_filters=100,
                 hidden_dim=320, num_layers=2, dropout=0.4,
                 max_seq_len=512, rope_base=10000.0):
        super().__init__()
        self.word_embed = nn.Embedding(vocab_size, word_dim, padding_idx=0)
        self.char_embed = nn.Embedding(char_vocab_size, char_dim, padding_idx=0)
        self.char_cnn = nn.Conv1d(in_channels=char_dim, out_channels=cnn_filters,
                                  kernel_size=3, padding=1)

        feature_dim = word_dim + cnn_filters
        assert feature_dim % 2 == 0, f"feature_dim={feature_dim} must be even for RoPE"

        self.rotary = RotaryEmbedding(dim=feature_dim, max_seq_len=max_seq_len, base=rope_base)
        self.lstm = nn.LSTM(feature_dim, hidden_dim // 2, num_layers=num_layers,
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

        features = torch.cat([w_emb, c_features], dim=2)
        features = self.rotary(features)          # ← RoPE applied here

        lstm_in = self.dropout(features)
        lstm_out, _ = self.lstm(lstm_in)
        emissions = self.hidden2tag(lstm_out)

        if labels is not None:
            return -self.crf(emissions, tags=labels, mask=mask, reduction='mean')
        return self.crf.decode(emissions, mask=mask), emissions


# ==========================================
# Address Parser
# ==========================================
class HKAddressParserBiLSTM:
    def __init__(self, model_path, hidden_dim=320, num_layers=2, dropout=0.4,
                 conf_threshold=0.50):
        """
        model_path : folder that contains
                     - vocabs.json
                     - best_by_logic.bin  (preferred)  or  pytorch_model.bin
        hidden_dim / num_layers / dropout : must match the trial you want to load
        conf_threshold : if split_conf < this value, emit a low-confidence warning
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.device = self._get_emptiest_gpu_safely()
        print(f"DEBUG: Using Device → {self.device}")

        vocab_path = os.path.join(model_path, "vocabs.json")
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"❌ vocabs.json not found in {model_path}")

        with open(vocab_path, "r", encoding="utf-8") as f:
            vocabs = json.load(f)
        self.w2i = vocabs["w2i"]
        self.c2i = vocabs["c2i"]
        self.t2i = vocabs["t2i"]
        self.idx2tag = {int(v): k for k, v in self.t2i.items()}

        self.model = BiLSTM_CNN_CRF(
            vocab_size=len(self.w2i),
            char_vocab_size=len(self.c2i),
            num_tags=len(self.t2i),
            word_dim=WORD_EMBED_DIM,
            char_dim=CHAR_EMBED_DIM,
            cnn_filters=CHAR_CNN_FILTERS,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            max_seq_len=MAX_SEQ_LEN_ROPE,
            rope_base=ROPE_BASE
        )

        # Prefer the file produced by the new training script
        weights_path = os.path.join(model_path, "best_by_logic.bin")
        if not os.path.exists(weights_path):
            weights_path = os.path.join(model_path, "pytorch_model.bin")

        if os.path.exists(weights_path):
            self._load_weights_with_progress(weights_path)
        else:
            raise FileNotFoundError(f"❌ No weights found in {model_path}")

        self.model.to(self.device)
        self.model.eval()

    def _load_weights_with_progress(self, weights_path):
        print(f"📦 Loading weights from {weights_path} ...")
        state_dict = torch.load(weights_path, map_location="cpu")

        # Handle both pure state_dict and full checkpoint dict
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

    def _tokenize_text(self, input_text):
        return [m.group() for m in re.finditer(
            r'[a-zA-Z]+|[0-9]+|[\u4e00-\u9fff]|[^\s]', input_text)]

    def _prepare_batch(self, batch_texts):
        batch_tokens = [self._tokenize_text(t) for t in batch_texts]
        max_seq_len = max((len(t) for t in batch_tokens), default=1)
        max_word_len = max((len(c) for t in batch_tokens for c in t), default=1)

        b_words, b_chars, b_masks = [], [], []
        for tokens in batch_tokens:
            seq_len = len(tokens)
            word_ids = [self.w2i.get(t, self.w2i.get("<UNK>", 1)) for t in tokens]
            char_ids_list = [[self.c2i.get(c, self.c2i.get("<UNK>", 1)) for c in tok]
                             for tok in tokens]

            b_words.append(word_ids + [0] * (max_seq_len - seq_len))
            b_masks.append([True] * seq_len + [False] * (max_seq_len - seq_len))

            padded = [ch + [0] * (max_word_len - len(ch)) for ch in char_ids_list]
            padded += [[0] * max_word_len] * (max_seq_len - seq_len)
            b_chars.append(padded)

        return (torch.tensor(b_words, dtype=torch.long),
                torch.tensor(b_chars, dtype=torch.long),
                torch.tensor(b_masks, dtype=torch.bool),
                batch_tokens)

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

    # ------------------------------------------------------------------
    # EXACT _split_address from the V7 training / evaluation script
    # + robust order-reversal / interleaving detection via group sequence
    # ------------------------------------------------------------------
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
                pass
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

        # ------------------------------------------------------------------
        # Robust is_reversed detection (no dependence on character offsets)
        # For Chinese we expect the resolved group sequence to be:
        #     (macro)* (micro)*
        # For English we expect:
        #     (micro)* (macro)*
        # Any inversion or interleaving (a "second" group appearing before
        # a later "first" group) is flagged as reversed / non-simple split.
        # This correctly catches cases such as:
        #   天爾天7座50樓h9 啟德協調道62號
        #   → Line1 (macro) containing both early + late tokens
        #   → Line2 (micro) containing the middle tokens
        # ------------------------------------------------------------------
        is_reversed = False
        if is_chinese:
            # expected order of groups: macro then micro
            seen_second = False
            for g in resolved_groups:
                if g == "micro":
                    seen_second = True
                elif g == "macro" and seen_second:
                    is_reversed = True
                    break
        else:
            # expected order of groups: micro then macro
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

            word_ids, char_ids, masks, batch_tokens = self._prepare_batch(full_addresses)
            word_ids = word_ids.to(self.device)
            char_ids = char_ids.to(self.device)
            masks = masks.to(self.device)

            try:
                with torch.no_grad():
                    pred_ids_batch, emissions_batch = self.model(word_ids, char_ids, masks)
                    batch_probs = torch.softmax(emissions_batch, dim=-1).cpu()
            except Exception as e:
                all_results.extend([("", "", "", {}, {}, 0.0, [], f"ERROR: {e}")] * len(batch_pairs))
                continue

            for idx, address_str in enumerate(full_addresses):
                if not address_str:
                    all_results.append(("", "", "", {}, {}, 0.0, [], "EMPTY_INPUT"))
                    continue

                pred_ids = pred_ids_batch[idx]
                probs = batch_probs[idx]
                tokens = batch_tokens[idx]

                char_tags = ["O"] * len(address_str)
                char_confs = [0.0] * len(address_str)
                start_indices = [m.start() for m in re.finditer(
                    r'[a-zA-Z]+|[0-9]+|[\u4e00-\u9fff]|[^\s]', address_str)]

                for j, tag_id in enumerate(pred_ids):
                    if j >= len(start_indices):
                        break
                    start_idx = start_indices[j]
                    tag = self.idx2tag[int(tag_id)]
                    conf = probs[j, int(tag_id)].item()
                    for c in range(start_idx, min(start_idx + len(tokens[j]), len(char_tags))):
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
                        "conf": conf,
                        "start": start_idx,
                    })

                extracted_raw, conf_raw = self._extract_3d_components(parsed_entities)
                extracted_mapped, conf_mapped = self._map_extracted_to_standard_keys(
                    extracted_raw, conf_raw)

                line1, line2, split_conf, used_keys, is_reversed = self._split_address(
                    extracted_mapped, address_str, parsed_entities, conf_mapped)

                # Build status: SUCCESS or one/more WARNINGs / ALERTs
                # (exact logic retained for compatibility with other modules)
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
    # Example: load one of the finished trials from the grid search
    # Change the path and the hyper-parameters to match the trial you want
    MODEL_DIR = "bilstm_v7"

    print(f"Loading model from {MODEL_DIR}...")
    parser = HKAddressParserBiLSTM(
        model_path=MODEL_DIR,
    )

    test_cases = [
        ("1M 16 lung sum avenue sheung shui north N.T", ""),
        ("27LD, Block 5, Hemera, Lohas Park, Lohas Road 1, TKO, HK", ""),
        ("21A / 5th Fl., Metroplaza Tower A, Nº 223 Hing Fong Road, Kwai Fong, N.T.", ""),
        ("長洲東灣東堤小築彌敦道一百二十三12座H室5樓", ""),
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
        print(f"Conf     : {{{', '.join(f'{k}: {v:.3f}' for k,v in confs.items())}}}")
        print(f"Logic keys: {used_keys}  |  Split conf: {overall:.4f}")
        print(f"status   : {status}")
        print("-" * 70)
