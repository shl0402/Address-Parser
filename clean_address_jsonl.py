
#!/usr/bin/env python3
"""
Clean + Augment Hong Kong address JSONL data.

1. Cleaning
   - Moves block / phase tokens that were incorrectly placed inside
     building_name or estate_name into the correct dedicated fields,
     then strips them from the name fields.
   - Fixes street_name that incorrectly includes building_number
     (common data error): strips the number from street_name, or
     extracts a trailing building number from street_name into the
     building_number field when the latter is empty.
   - Ensures the structured output forms a strict non-overlapping
     partition of the input text (required for BIO NER).  Records
     that still contain repeated / overlapping entity values after
     cleaning are reported and removed from the output JSONL.

2. Data Augmentation (optional, controlled by variables below)
   - With probability AUGMENT_PROB, randomly wrap 1 or more address
     components with parentheses.
   - BOTH the input string AND the corresponding fields in the structured
     output are updated, so labels stay consistent with the text.
     Example (correct):
       input : "(ROOM 06), (64 Floor), ..., (143) BONHAM STRAND, ..."
       flat  : "(ROOM 06)"
       floor : "(64 Floor)"
       building_number : "(143)"
   - Pattern distribution (given that we decided to augment):
       ONE_TAG_PROB   : wrap a single component          e.g. ...(5樓)...
       TWO_TAG_PROB   : wrap two components together     e.g. ...(5樓 A室)...
       remainder      : wrap 3 components or a larger span


Usage:
    python clean_address_jsonl.py [--input-dir DIR] [--output-dir DIR]
                                  [--files train test validation]
                                  [--no-augment] [--seed 42]

Produces:
    <name>_cleaned.jsonl          (cleaned + optionally augmented; invalid partitions removed)
    cleaning_report.txt
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ===========================================================================
#  >>>  EASY-TO-CHANGE AUGMENTATION VARIABLES  <<<
# ===========================================================================

# Probability that a record will receive any parentheses augmentation
AUGMENT_PROB = 0.02

# Given that we augment, probability of each pattern style
ONE_TAG_PROB = 0.70          # wrap exactly one component
TWO_TAG_PROB = 0.20          # wrap two components together  → (floor flat) / (street estate) ...
# remaining 0.10             → three components or a broader span

# Which fields are allowed to be wrapped (order = preference when sampling)
# line1 fields first, then line2
WRAPPABLE_FIELDS = [
    # line1
    "flat", "floor", "block", "building_name", "phase", "estate_name",
    # line2
    "building_number", "street_name", "village_name", "sub_district", "district", "region",
]

# Minimum length of a component value to be considered for wrapping
MIN_COMPONENT_LEN = 1

# ===========================================================================


# ---------------------------------------------------------------------------
# Cleaning patterns
# ---------------------------------------------------------------------------

PHASE_PATTERNS = [
    re.compile(r"(\([0-9一二三四五六七八九十]+[区區]\))", re.UNICODE),
    re.compile(r"(第?[0-9一二三四五六七八九十]+期)", re.UNICODE),
    re.compile(r"((?:Phase|PHASE)\s*[IVX0-9]+)", re.IGNORECASE),
]

BLOCK_PATTERNS = [
    re.compile(
        r"((?:第?[0-9一二三四五六七八九十百千零]+|[A-Za-z][A-Za-z0-9]{0,3}|[0-9]{1,3})座)",
        re.UNICODE,
    ),
    re.compile(r"((?:Block|Blk\.?)\s*[A-Za-z0-9]+)", re.IGNORECASE),
    re.compile(r"((?:HOUSE|HSE|House|Hse)\s+[0-9A-Za-z]+)", re.IGNORECASE),
    re.compile(r"(第?[0-9A-Za-z]+洋房)", re.UNICODE),
    re.compile(r"([0-9A-Za-z]{1,4}屋)", re.UNICODE),
]

# Trailing building-number pattern (Arabic / Chinese digits + optional 號)
BUILDING_NUMBER_TAIL = re.compile(
    r"([0-9０-９一二三四五六七八九十百千零]+號?)\s*$",
    re.UNICODE,
)


def _find_best(text: str, patterns: List[re.Pattern]) -> Optional[Tuple[str, int, int]]:
    """Prefer rightmost, then longest match."""
    if not text:
        return None
    best = None
    for pat in patterns:
        for m in pat.finditer(text):
            token, start, end = m.group(1), m.start(), m.end()
            length = end - start
            if best is None or end > best[2] or (end == best[2] and length > best[3]):
                best = (token, start, end, length)
    if best is None:
        return None
    return best[0], best[1], best[2]


def extract_phase(text: str) -> Optional[Tuple[str, str]]:
    found = _find_best(text, PHASE_PATTERNS)
    if not found:
        return None
    token, start, end = found
    before = text[:start].rstrip(" -–—/")
    after = text[end:].lstrip(" -–—/")
    cleaned = re.sub(r"\s{2,}", " ", (before + " " + after).strip())
    return cleaned, token.strip()


def extract_block(text: str) -> Optional[Tuple[str, str]]:
    found = _find_best(text, BLOCK_PATTERNS)
    if not found:
        return None
    token, start, end = found
    before = text[:start].rstrip(" -–—/")
    after = text[end:].lstrip(" -–—/")
    cleaned = re.sub(r"\s{2,}", " ", (before + " " + after).strip())
    return cleaned, token.strip()


def _strip_or_extract_building_number(
    street: str, bnum: str
) -> Tuple[str, str, Optional[str]]:
    """
    If building_number is present inside street_name, strip it out of street.
    If building_number is empty but street ends with a number(+號), extract it.
    Returns (new_street, new_bnum, change_description or None).
    """
    street = (street or "").strip()
    bnum = (bnum or "").strip()

    if not street:
        return street, bnum, None

    # Case 1: building_number already set and appears inside street → strip
    if bnum and bnum in street:
        # Prefer suffix (most common: "...街29號")
        if street.endswith(bnum):
            new_street = street[: -len(bnum)].rstrip(" -–—/，,")
            new_street = re.sub(r"\s{2,}", " ", new_street).strip()
            if new_street != street:
                return new_street, bnum, f"stripped '{bnum}' from street_name (suffix)"
        else:
            # Remove rightmost occurrence
            idx = street.rfind(bnum)
            if idx >= 0:
                before = street[:idx].rstrip(" -–—/，,")
                after = street[idx + len(bnum) :].lstrip(" -–—/，,")
                new_street = re.sub(r"\s{2,}", " ", (before + " " + after).strip()).strip()
                if new_street != street:
                    return new_street, bnum, f"stripped '{bnum}' from street_name"

    # Case 2: no building_number, try to extract a trailing number from street
    if not bnum:
        m = BUILDING_NUMBER_TAIL.search(street)
        if m:
            extracted = m.group(1).strip()
            new_street = street[: m.start()].rstrip(" -–—/，,")
            new_street = re.sub(r"\s{2,}", " ", new_street).strip()
            if new_street and extracted:
                return new_street, extracted, f"extracted building_number '{extracted}' ← street_name"

    return street, bnum, None


def clean_record(rec: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Clean one record. Returns (new_record, list_of_change_descriptions)."""
    changes: List[str] = []
    out = rec.get("output")
    if not isinstance(out, dict):
        return rec, changes

    new_line1 = dict(out.get("line1") or {})
    new_line2 = dict(out.get("line2") or {})

    bname = (new_line1.get("building_name") or "").strip()
    ename = (new_line1.get("estate_name") or "").strip()
    block = (new_line1.get("block") or "").strip()
    phase = (new_line1.get("phase") or "").strip()

    # 1. phase from estate_name
    if ename and not phase:
        result = extract_phase(ename)
        if result:
            new_ename, extracted = result
            if new_ename != ename and extracted:
                new_line1["estate_name"] = new_ename
                new_line1["phase"] = extracted
                phase = extracted
                ename = new_ename
                changes.append(f"phase '{extracted}' ← estate_name")

    # 2. phase from building_name
    if bname and not phase:
        result = extract_phase(bname)
        if result:
            new_bname, extracted = result
            if new_bname != bname and extracted:
                new_line1["building_name"] = new_bname
                new_line1["phase"] = extracted
                phase = extracted
                bname = new_bname
                changes.append(f"phase '{extracted}' ← building_name")

    # 3. block from building_name
    if bname and not block:
        result = extract_block(bname)
        if result:
            new_bname, extracted = result
            if new_bname != bname and extracted:
                new_line1["building_name"] = new_bname
                new_line1["block"] = extracted
                block = extracted
                bname = new_bname
                changes.append(f"block '{extracted}' ← building_name")

    # 4. block from estate_name
    if ename and not block:
        result = extract_block(ename)
        if result:
            new_ename, extracted = result
            if new_ename != ename and extracted:
                new_line1["estate_name"] = new_ename
                new_line1["block"] = extracted
                block = extracted
                ename = new_ename
                changes.append(f"block '{extracted}' ← estate_name")

    # 5. clear building_name if it became identical to estate_name
    bname = (new_line1.get("building_name") or "").strip()
    ename = (new_line1.get("estate_name") or "").strip()
    if bname and ename and bname == ename:
        new_line1["building_name"] = ""
        changes.append("cleared building_name (duplicate of estate_name)")

    # 6. Fix street_name that embeds building_number (or extract number when missing)
    street = (new_line2.get("street_name") or "").strip()
    bnum = (new_line2.get("building_number") or "").strip()
    new_street, new_bnum, desc = _strip_or_extract_building_number(street, bnum)
    if desc:
        new_line2["street_name"] = new_street
        new_line2["building_number"] = new_bnum
        changes.append(desc)

    # Final strip of all string fields
    for key in ("building_name", "estate_name", "block", "phase", "flat", "floor"):
        if key in new_line1 and isinstance(new_line1[key], str):
            new_line1[key] = new_line1[key].strip()
    for key in (
        "building_number",
        "street_name",
        "village_name",
        "sub_district",
        "district",
        "region",
    ):
        if key in new_line2 and isinstance(new_line2[key], str):
            new_line2[key] = new_line2[key].strip()

    new_out = dict(out)
    new_out["line1"] = new_line1
    new_out["line2"] = new_line2
    new_rec = dict(rec)
    new_rec["output"] = new_out
    return new_rec, changes


# ---------------------------------------------------------------------------
# Partition validation (required for BIO NER)
# ---------------------------------------------------------------------------

def validate_partition(rec: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Verify that every non-empty label value is an exact substring of the input
    and that the chosen spans do not overlap.  Longer values are assigned
    first so that a container (e.g. street containing a number) will claim
    the span and cause the nested value to fail unless it was already cleaned.

    Returns (is_valid, reason).  reason is empty on success.
    """
    input_text = rec.get("input") or ""
    if not input_text:
        return False, "empty input"

    out = rec.get("output") or {}
    line1 = out.get("line1") or {}
    line2 = out.get("line2") or {}

    components: List[Tuple[str, str]] = []
    for section, d in (("line1", line1), ("line2", line2)):
        if not isinstance(d, dict):
            continue
        for k, v in d.items():
            v = (v or "").strip()
            if v:
                components.append((f"{section}.{k}", v))

    if not components:
        return True, ""

    # Longer first → containers claim span before nested pieces
    comps_sorted = sorted(components, key=lambda x: (-len(x[1]), x[0]))

    used: List[Tuple[int, int]] = []  # (start, end) exclusive

    for fname, val in comps_sorted:
        # Collect all exact occurrences
        starts: List[int] = []
        pos = 0
        while True:
            idx = input_text.find(val, pos)
            if idx < 0:
                break
            starts.append(idx)
            pos = idx + 1

        if not starts:
            return False, f"value '{val}' ({fname}) not found in input"

        # Pick the first occurrence that does not overlap any already used span
        chosen = None
        for s in starts:
            e = s + len(val)
            overlaps = any(max(s, us) < min(e, ue) for us, ue in used)
            if not overlaps:
                chosen = (s, e)
                break

        if chosen is None:
            return False, f"no non-overlapping span left for '{val}' ({fname})"

        used.append(chosen)

    return True, ""


# ---------------------------------------------------------------------------
# Augmentation helpers
# ---------------------------------------------------------------------------

def _collect_components(rec: Dict[str, Any]) -> List[Tuple[str, str]]:
    """
    Return list of (field_name, value) for non-empty wrappable components.
    Values are stripped and must meet MIN_COMPONENT_LEN.
    """
    out = rec.get("output") or {}
    line1 = out.get("line1") or {}
    line2 = out.get("line2") or {}
    comps = []
    for field in WRAPPABLE_FIELDS:
        val = (line1.get(field) or line2.get(field) or "").strip()
        if len(val) >= MIN_COMPONENT_LEN:
            comps.append((field, val))
    return comps


def _find_span(text: str, value: str) -> Optional[Tuple[int, int]]:
    """
    Find the first occurrence of `value` inside `text` (case-insensitive for
    English, exact for Chinese). Returns (start, end) or None.
    """
    if not value:
        return None
    # Try exact first
    idx = text.find(value)
    if idx >= 0:
        return idx, idx + len(value)
    # Case-insensitive fallback (useful for English addresses)
    lower_text = text.lower()
    lower_val = value.lower()
    idx = lower_text.find(lower_val)
    if idx >= 0:
        return idx, idx + len(value)
    return None


def _wrap_spans(text: str, spans: List[Tuple[int, int]]) -> str:
    """
    Insert parentheses around the given character spans.
    Spans must be non-overlapping and sorted by start position.
    We insert from right to left so indices stay valid.
    """
    spans = sorted(spans, key=lambda s: s[0], reverse=True)
    for start, end in spans:
        # Avoid double-wrapping if already inside parentheses
        if start > 0 and text[start - 1] == "(" and end < len(text) and text[end] == ")":
            continue
        text = text[:start] + "(" + text[start:end] + ")" + text[end:]
    return text


def augment_record(rec: Dict[str, Any], rng: random.Random) -> Tuple[Dict[str, Any], str]:
    """
    Possibly augment the record by wrapping components with parentheses.

    Both the *input* string AND the corresponding fields in the structured
    *output* are updated so that labels stay consistent with the text.

    Example of correct behaviour:
        input : "(ROOM 06), (64 Floor), CHAO'S BUILDING, (143) BONHAM STRAND, ..."
        flat  : "(ROOM 06)"
        floor : "(64 Floor)"
        building_number : "(143)"

    Returns (new_record, description).
    description is empty when no augmentation happened.
    """
    original_input = rec.get("input") or ""
    if not original_input or rng.random() >= AUGMENT_PROB:
        return rec, ""

    comps = _collect_components(rec)
    if not comps:
        return rec, ""

    # Decide how many tags to wrap
    r = rng.random()
    if r < ONE_TAG_PROB:
        n = 1
        style = "one_tag"
    elif r < ONE_TAG_PROB + TWO_TAG_PROB:
        n = 2
        style = "two_tag"
    else:
        n = 3
        style = "multi_tag"

    n = min(n, len(comps))
    chosen = rng.sample(comps, n)

    # Locate their spans in the input and record which fields we successfully found
    spans: List[Tuple[int, int]] = []
    field_to_value: Dict[str, str] = {}  # field -> original value that was wrapped
    for field, value in chosen:
        # Skip values that already start/end with parentheses
        if value.startswith("(") and value.endswith(")"):
            continue
        span = _find_span(original_input, value)
        if span is not None:
            spans.append(span)
            field_to_value[field] = value

    if not spans:
        return rec, ""

    # Merge overlapping / adjacent spans so we produce clean parentheses
    spans = sorted(spans, key=lambda s: s[0])
    merged = [list(spans[0])]
    for s, e in spans[1:]:
        if s <= merged[-1][1] + 1:  # overlapping or adjacent
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])

    new_input = _wrap_spans(original_input, [tuple(m) for m in merged])

    # ---- Update the structured output so parentheses appear inside the tags ----
    new_rec = dict(rec)
    new_out = dict(rec.get("output") or {})
    new_line1 = dict(new_out.get("line1") or {})
    new_line2 = dict(new_out.get("line2") or {})

    LINE1_FIELDS = {"flat", "floor", "block", "building_name", "phase", "estate_name"}
    LINE2_FIELDS = {"building_number", "street_name", "village_name", "sub_district", "district", "region"}

    updated_fields = []
    for field, original_val in field_to_value.items():
        wrapped_val = f"({original_val})"
        if field in LINE1_FIELDS:
            # Only update if the current value still matches what we wrapped
            if (new_line1.get(field) or "").strip() == original_val:
                new_line1[field] = wrapped_val
                updated_fields.append(field)
        elif field in LINE2_FIELDS:
            if (new_line2.get(field) or "").strip() == original_val:
                new_line2[field] = wrapped_val
                updated_fields.append(field)

    new_out["line1"] = new_line1
    new_out["line2"] = new_line2
    new_rec["output"] = new_out
    new_rec["input"] = new_input

    desc = f"augment({style}): wrapped {updated_fields}"
    return new_rec, desc



# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_file(
    src: Path,
    dst: Path,
    do_augment: bool,
    seed: int,
    max_samples: int = 30,
) -> Dict[str, Any]:
    stats = Counter()
    sample_changes: List[Dict[str, Any]] = []
    sample_removed: List[Dict[str, Any]] = []
    rng = random.Random(seed)

    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                stats["json_error"] += 1
                fout.write(line + "\n")
                continue

            stats["total"] += 1

            # ---- 1. Cleaning ----
            cleaned, clean_changes = clean_record(rec)

            # ---- 2. Partition validation (strict non-overlapping spans) ----
            is_valid, reason = validate_partition(cleaned)
            if not is_valid:
                stats["removed"] += 1
                if len(sample_removed) < max_samples:
                    l1 = cleaned["output"].get("line1") or {}
                    l2 = cleaned["output"].get("line2") or {}
                    sample_removed.append(
                        {
                            "line": line_no,
                            "input": (cleaned.get("input") or "")[:160],
                            "reason": reason,
                            "labels": {
                                "flat": l1.get("flat", ""),
                                "floor": l1.get("floor", ""),
                                "block": l1.get("block", ""),
                                "building_name": l1.get("building_name", ""),
                                "phase": l1.get("phase", ""),
                                "estate_name": l1.get("estate_name", ""),
                                "building_number": l2.get("building_number", ""),
                                "street_name": l2.get("street_name", ""),
                                "district": l2.get("district", ""),
                                "region": l2.get("region", ""),
                            },
                            "clean_ops": clean_changes,
                        }
                    )
                continue

            # ---- 3. Augmentation (updates both input AND the matching output tags) ----
            aug_desc = ""
            if do_augment:
                cleaned, aug_desc = augment_record(cleaned, rng)
                if aug_desc:
                    # Re-validate after augmentation (should almost always pass)
                    is_valid2, reason2 = validate_partition(cleaned)
                    if not is_valid2:
                        stats["removed"] += 1
                        if len(sample_removed) < max_samples:
                            sample_removed.append(
                                {
                                    "line": line_no,
                                    "input": (cleaned.get("input") or "")[:160],
                                    "reason": f"after_augment: {reason2}",
                                    "labels": {},
                                    "clean_ops": clean_changes,
                                    "aug_ops": aug_desc,
                                }
                            )
                        continue
                    stats["augmented"] += 1

            if clean_changes:
                stats["changed"] += 1
                for c in clean_changes:
                    if "phase" in c:
                        stats["moved_phase"] += 1
                    if "block" in c:
                        stats["moved_block"] += 1
                    if "cleared building_name" in c:
                        stats["cleared_dup_bname"] += 1
                    if "street_name" in c or "building_number" in c:
                        stats["fixed_street_bnum"] += 1

            if clean_changes or aug_desc:
                if len(sample_changes) < max_samples:
                    # richer sample that also shows flat / floor / building_number
                    l1_before = rec["output"].get("line1") or {}
                    l2_before = rec["output"].get("line2") or {}
                    l1_after = cleaned["output"].get("line1") or {}
                    l2_after = cleaned["output"].get("line2") or {}
                    sample_changes.append(
                        {
                            "line": line_no,
                            "input_before": (rec.get("input") or "")[:120],
                            "input_after": (cleaned.get("input") or "")[:120],
                            "before": {
                                "flat": l1_before.get("flat", ""),
                                "floor": l1_before.get("floor", ""),
                                "block": l1_before.get("block", ""),
                                "building_name": l1_before.get("building_name", ""),
                                "phase": l1_before.get("phase", ""),
                                "estate_name": l1_before.get("estate_name", ""),
                                "building_number": l2_before.get("building_number", ""),
                                "street_name": l2_before.get("street_name", ""),
                            },
                            "after": {
                                "flat": l1_after.get("flat", ""),
                                "floor": l1_after.get("floor", ""),
                                "block": l1_after.get("block", ""),
                                "building_name": l1_after.get("building_name", ""),
                                "phase": l1_after.get("phase", ""),
                                "estate_name": l1_after.get("estate_name", ""),
                                "building_number": l2_after.get("building_number", ""),
                                "street_name": l2_after.get("street_name", ""),
                            },
                            "clean_ops": clean_changes,
                            "aug_ops": aug_desc,
                        }
                    )

            fout.write(json.dumps(cleaned, ensure_ascii=False) + "\n")

    return {
        "stats": dict(stats),
        "samples": sample_changes,
        "removed_samples": sample_removed,
        "src": str(src),
        "dst": str(dst),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean block/phase/street fields, enforce non-overlapping partitions for BIO NER, and optionally augment inputs with parentheses"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("./data"),
        help="Directory containing the original *.jsonl files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./data"),
        help="Directory to write *_cleaned.jsonl and the report",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=["train", "test", "validation"],
        help="Base names to process (without .jsonl)",
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable parentheses augmentation (cleaning only)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible augmentation",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=40,
        help="How many sample changes / removals to keep in the report",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    do_augment = not args.no_augment

    report_lines: List[str] = []
    report_lines.append("=" * 72)
    report_lines.append("Address JSONL Cleaning + Augmentation Report")
    report_lines.append("=" * 72)
    report_lines.append(f"AUGMENT_PROB     = {AUGMENT_PROB}")
    report_lines.append(f"ONE_TAG_PROB     = {ONE_TAG_PROB}")
    report_lines.append(f"TWO_TAG_PROB     = {TWO_TAG_PROB}")
    report_lines.append(f"Augmentation     = {'ON' if do_augment else 'OFF'}")
    report_lines.append(f"Random seed      = {args.seed}")
    report_lines.append("")

    all_results = []
    for name in args.files:
        src = args.input_dir / f"{name}.jsonl"
        if not src.exists():
            report_lines.append(f"[SKIP] {src} does not exist")
            print(f"[SKIP] {src} not found")
            continue

        dst = args.output_dir / f"{name}_cleaned.jsonl"
        print(f"Processing {src} → {dst} (augment={do_augment}) ...")
        result = process_file(
            src, dst, do_augment=do_augment, seed=args.seed, max_samples=args.max_samples
        )
        all_results.append(result)

        st = result["stats"]
        report_lines.append(f"File: {name}.jsonl")
        report_lines.append(f"  Total records        : {st.get('total', 0)}")
        report_lines.append(f"  Records cleaned      : {st.get('changed', 0)}")
        report_lines.append(f"  Block tokens moved   : {st.get('moved_block', 0)}")
        report_lines.append(f"  Phase tokens moved   : {st.get('moved_phase', 0)}")
        report_lines.append(f"  Cleared dup bname    : {st.get('cleared_dup_bname', 0)}")
        report_lines.append(f"  Fixed street/bnum    : {st.get('fixed_street_bnum', 0)}")
        report_lines.append(f"  Records augmented    : {st.get('augmented', 0)}")
        report_lines.append(f"  Records removed      : {st.get('removed', 0)}  (partition / repeat violations)")
        report_lines.append(f"  JSON errors          : {st.get('json_error', 0)}")
        report_lines.append(f"  Output               : {result['dst']}")
        report_lines.append("")

        if result["samples"]:
            report_lines.append(f"  --- Sample changes (first {len(result['samples'])}) ---")
            for i, s in enumerate(result["samples"], 1):
                report_lines.append(f"  [{i}] line {s['line']}")
                report_lines.append(f"      input before : {s['input_before']}")
                report_lines.append(f"      input after  : {s['input_after']}")
                report_lines.append(f"      labels before: {s['before']}")
                report_lines.append(f"      labels after : {s['after']}")
                report_lines.append(f"      clean ops    : {s['clean_ops']}")
                report_lines.append(f"      aug ops      : {s['aug_ops']}")
                report_lines.append("")
        if result.get("removed_samples"):
            report_lines.append(f"  --- Sample REMOVED records (first {len(result['removed_samples'])}) ---")
            for i, s in enumerate(result["removed_samples"], 1):
                report_lines.append(f"  [{i}] line {s['line']}")
                report_lines.append(f"      input        : {s['input']}")
                report_lines.append(f"      reason       : {s['reason']}")
                report_lines.append(f"      labels       : {s.get('labels', {})}")
                report_lines.append(f"      clean ops    : {s.get('clean_ops', [])}")
                if s.get("aug_ops"):
                    report_lines.append(f"      aug ops      : {s['aug_ops']}")
                report_lines.append("")
        report_lines.append("-" * 72)
        report_lines.append("")

    report_path = args.output_dir / "cleaning_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\nReport written to {report_path}")

    print("\n=== Quick summary ===")
    for r in all_results:
        st = r["stats"]
        print(
            f"{Path(r['src']).name}: "
            f"cleaned={st.get('changed', 0)}/{st.get('total', 0)}, "
            f"fixed_street={st.get('fixed_street_bnum', 0)}, "
            f"augmented={st.get('augmented', 0)}, "
            f"removed={st.get('removed', 0)}, "
            f"block={st.get('moved_block', 0)}, phase={st.get('moved_phase', 0)}"
        )


if __name__ == "__main__":
    main()