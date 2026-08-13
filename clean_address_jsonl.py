#!/usr/bin/env python3
"""
Clean Hong Kong address JSONL data.

Moves block / phase tokens that were incorrectly placed inside
building_name or estate_name into the correct dedicated fields,
then strips them from the name fields.

Usage:
    python clean_address_jsonl.py [--input-dir DIR] [--output-dir DIR] [--files train test validation]

Produces:
    <name>_cleaned.jsonl
    cleaning_report.txt   (summary + sample changes)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Patterns (ordered from more specific to less specific)
# ---------------------------------------------------------------------------

# Phase patterns – extract first so compounds like "第一期第10座" work
PHASE_PATTERNS = [
    # (二区) / (二區)
    re.compile(r"(\([0-9一二三四五六七八九十]+[区區]\))", re.UNICODE),
    # 第X期 / 第二期 / 一期 / 三期
    re.compile(r"(第?[0-9一二三四五六七八九十]+期)", re.UNICODE),
    # Phase II / Phase 3 / PHASE VI
    re.compile(r"((?:Phase|PHASE)\s*[IVX0-9]+)", re.IGNORECASE),
]

# Block patterns – more specific / longer matches first
BLOCK_PATTERNS = [
    # Combined Chinese / alphanumeric seat: 第11座, 第二十座, 六座, E31座, D5座, F座, A4座, H4座
    re.compile(r"((?:第?[0-9一二三四五六七八九十百千零]+|[A-Za-z][A-Za-z0-9]{0,3}|[0-9]{1,3})座)", re.UNICODE),
    # English Block / Blk
    re.compile(r"((?:Block|Blk\.?)\s*[A-Za-z0-9]+)", re.IGNORECASE),
    # HOUSE / HSE + number (require space to avoid matching "warehouse" etc.)
    re.compile(r"((?:HOUSE|HSE|House|Hse)\s+[0-9A-Za-z]+)", re.IGNORECASE),
    # 洋房
    re.compile(r"(第?[0-9A-Za-z]+洋房)", re.UNICODE),
    # short 屋 (C10屋, 15屋, 9屋) – keep relatively strict
    re.compile(r"([0-9A-Za-z]{1,4}屋)", re.UNICODE),
]


def _find_first(text: str, patterns: List[re.Pattern]) -> Optional[Tuple[str, int, int]]:
    """Return (matched_string, start, end). Prefer rightmost, then longest match."""
    if not text:
        return None
    best = None  # (token, start, end, length)
    for pat in patterns:
        for m in pat.finditer(text):
            token = m.group(1)
            start, end = m.start(), m.end()
            length = end - start
            if best is None:
                best = (token, start, end, length)
            else:
                # Prefer later end; if same end, prefer longer match
                if end > best[2] or (end == best[2] and length > best[3]):
                    best = (token, start, end, length)
    if best is None:
        return None
    return best[0], best[1], best[2]


def extract_phase(text: str) -> Optional[Tuple[str, str]]:
    """
    Returns (cleaned_text, extracted_phase) or None if nothing extracted.
    Removes the phase token and surrounding whitespace / punctuation.
    """
    found = _find_first(text, PHASE_PATTERNS)
    if not found:
        return None
    token, start, end = found
    # Remove token + any leading/trailing spaces or punctuation that glued it
    before = text[:start].rstrip(" -–—/")
    after = text[end:].lstrip(" -–—/")
    cleaned = (before + " " + after).strip()
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned, token.strip()


def extract_block(text: str) -> Optional[Tuple[str, str]]:
    """Same idea for block."""
    found = _find_first(text, BLOCK_PATTERNS)
    if not found:
        return None
    token, start, end = found
    before = text[:start].rstrip(" -–—/")
    after = text[end:].lstrip(" -–—/")
    cleaned = (before + " " + after).strip()
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned, token.strip()


def clean_record(rec: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """
    Clean one record in-place (returns a new dict) and a list of change descriptions.
    """
    changes: List[str] = []
    out = rec.get("output")
    if not isinstance(out, dict):
        return rec, changes

    line1 = out.get("line1")
    if not isinstance(line1, dict):
        return rec, changes

    # Work on copies
    new_line1 = dict(line1)
    bname = (new_line1.get("building_name") or "").strip()
    ename = (new_line1.get("estate_name") or "").strip()
    block = (new_line1.get("block") or "").strip()
    phase = (new_line1.get("phase") or "").strip()

    original = {
        "building_name": bname,
        "estate_name": ename,
        "block": block,
        "phase": phase,
    }

    # ------------------------------------------------------------------
    # 1. Extract phase from estate_name (most common for (二區) style)
    # ------------------------------------------------------------------
    if ename and not phase:
        result = extract_phase(ename)
        if result:
            new_ename, extracted = result
            # Only accept if the phase was not the entire name
            if new_ename != ename and extracted:
                new_line1["estate_name"] = new_ename
                new_line1["phase"] = extracted
                phase = extracted
                ename = new_ename
                changes.append(f"phase '{extracted}' ← estate_name")

    # ------------------------------------------------------------------
    # 2. Extract phase from building_name
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 3. Extract block from building_name (most common)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 4. Extract block from estate_name (rare)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 5. Post-clean: if building_name now equals estate_name, clear building_name
    #    (common after stripping "匡湖居21座" → "匡湖居" while estate is already "匡湖居")
    # ------------------------------------------------------------------
    bname = (new_line1.get("building_name") or "").strip()
    ename = (new_line1.get("estate_name") or "").strip()
    if bname and ename and bname == ename:
        new_line1["building_name"] = ""
        changes.append("cleared building_name (duplicate of estate_name)")

    # Also clear if building_name became empty after stripping
    if not bname:
        new_line1["building_name"] = ""

    # Final safety: never leave leading/trailing spaces
    for key in ("building_name", "estate_name", "block", "phase"):
        if key in new_line1 and isinstance(new_line1[key], str):
            new_line1[key] = new_line1[key].strip()

    # Rebuild output
    new_out = dict(out)
    new_out["line1"] = new_line1
    new_rec = dict(rec)
    new_rec["output"] = new_out

    return new_rec, changes


def process_file(
    src: Path,
    dst: Path,
    max_samples: int = 30,
) -> Dict[str, Any]:
    """Process one jsonl file, write cleaned version, return stats."""
    stats = Counter()
    sample_changes: List[Dict[str, Any]] = []

    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                stats["json_error"] += 1
                fout.write(line + "\n")
                continue

            stats["total"] += 1
            cleaned, changes = clean_record(rec)

            if changes:
                stats["changed"] += 1
                for c in changes:
                    if "phase" in c:
                        stats["moved_phase"] += 1
                    if "block" in c:
                        stats["moved_block"] += 1
                    if "cleared building_name" in c:
                        stats["cleared_dup_bname"] += 1

                if len(sample_changes) < max_samples:
                    sample_changes.append(
                        {
                            "line": line_no,
                            "input": (rec.get("input") or "")[:100],
                            "before": {
                                "building_name": rec["output"]["line1"].get("building_name", ""),
                                "estate_name": rec["output"]["line1"].get("estate_name", ""),
                                "block": rec["output"]["line1"].get("block", ""),
                                "phase": rec["output"]["line1"].get("phase", ""),
                            },
                            "after": {
                                "building_name": cleaned["output"]["line1"].get("building_name", ""),
                                "estate_name": cleaned["output"]["line1"].get("estate_name", ""),
                                "block": cleaned["output"]["line1"].get("block", ""),
                                "phase": cleaned["output"]["line1"].get("phase", ""),
                            },
                            "changes": changes,
                        }
                    )

            fout.write(json.dumps(cleaned, ensure_ascii=False) + "\n")

    return {
        "stats": dict(stats),
        "samples": sample_changes,
        "src": str(src),
        "dst": str(dst),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean block/phase fields in address JSONL files")
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
        "--max-samples",
        type=int,
        default=40,
        help="How many sample changes to keep in the report",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    report_lines: List[str] = []
    report_lines.append("=" * 72)
    report_lines.append("Address JSONL Cleaning Report")
    report_lines.append("=" * 72)
    report_lines.append("")

    for name in args.files:
        src = args.input_dir / f"{name}.jsonl"
        if not src.exists():
            report_lines.append(f"[SKIP] {src} does not exist")
            print(f"[SKIP] {src} not found")
            continue

        dst = args.output_dir / f"{name}_cleaned.jsonl"
        print(f"Processing {src} → {dst} ...")
        result = process_file(src, dst, max_samples=args.max_samples)
        all_results.append(result)

        st = result["stats"]
        report_lines.append(f"File: {name}.jsonl")
        report_lines.append(f"  Total records     : {st.get('total', 0)}")
        report_lines.append(f"  Records changed   : {st.get('changed', 0)}")
        report_lines.append(f"  Block tokens moved: {st.get('moved_block', 0)}")
        report_lines.append(f"  Phase tokens moved: {st.get('moved_phase', 0)}")
        report_lines.append(f"  Cleared dup bname : {st.get('cleared_dup_bname', 0)}")
        report_lines.append(f"  JSON errors       : {st.get('json_error', 0)}")
        report_lines.append(f"  Output            : {result['dst']}")
        report_lines.append("")

        if result["samples"]:
            report_lines.append(f"  --- Sample changes (first {len(result['samples'])}) ---")
            for i, s in enumerate(result["samples"], 1):
                report_lines.append(f"  [{i}] line {s['line']}")
                report_lines.append(f"      input : {s['input']}")
                report_lines.append(f"      before: {s['before']}")
                report_lines.append(f"      after : {s['after']}")
                report_lines.append(f"      ops   : {s['changes']}")
                report_lines.append("")
        report_lines.append("-" * 72)
        report_lines.append("")

    # Write report
    report_path = args.output_dir / "cleaning_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\nReport written to {report_path}")
    print("\n".join(report_lines[:80]))  # preview

    # Also print a quick summary to stdout
    print("\n=== Quick summary ===")
    for r in all_results:
        st = r["stats"]
        print(
            f"{Path(r['src']).name}: "
            f"{st.get('changed', 0)}/{st.get('total', 0)} changed "
            f"(block={st.get('moved_block', 0)}, phase={st.get('moved_phase', 0)})"
        )


if __name__ == "__main__":
    main()
