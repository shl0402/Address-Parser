#!/usr/bin/env python3
"""Reduce generated HK address JSONL rows to model-ready input/output pairs.

Only entity values visibly present in each input string are copied. Omitted
components therefore become empty strings instead of leaking canonical ALS
metadata into the training target.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, TextIO

LOGGER = logging.getLogger("compact_hk_address_extractor")
OUTPUT_FIELDS = (
    "flat",
    "floor",
    "block",
    "building_name",
    "phase",
    "estate_name",
    "building_number",
    "street_name",
    "village_name",
    "sub_district",
    "district",
    "region",
)
SEPARATOR_ONLY = re.compile(r"^[\s,，、;/／.·-]*$")

REGION_CODES = {
    "HK": "HK",
    "HONG KONG": "HK",
    "HONG KONG ISLAND": "HK",
    "香港": "HK",
    "香港島": "HK",
    "香港岛": "HK",
    "港島": "HK",
    "港岛": "HK",
    "香港區": "HK",
    "香港区": "HK",
    "KLN": "KLN",
    "KOWLOON": "KLN",
    "九龍": "KLN",
    "九龙": "KLN",
    "九龍區": "KLN",
    "九龙区": "KLN",
    "NT": "NT",
    "NEW TERRITORIES": "NT",
    "NORTH TERRITORIES": "NT",
    "新界": "NT",
    "新界區": "NT",
    "新界区": "NT",
}


class ExtractionError(RuntimeError):
    """Raised for invalid files, rows, or CLI settings."""


def as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def clean_text(value: Any) -> str:
    if value is None or isinstance(value, (Mapping, list, tuple)):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def load_tqdm(enabled: bool) -> Any:
    if not enabled:
        return None
    try:
        from tqdm.auto import tqdm  # type: ignore
    except ImportError as exc:
        raise ExtractionError(
            "Progress display needs tqdm. Install it with `pip install tqdm`, "
            "or use --no-progress."
        ) from exc
    return tqdm


def expand_splits(values: Sequence[str]) -> tuple[str, ...]:
    if "all" in values:
        return ("train", "validation", "test")
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return tuple(result)


def discover_inputs(
    supplied_paths: Sequence[Path], splits: Sequence[str], output_path: Path
) -> list[Path]:
    files: list[Path] = []
    wanted_splits = expand_splits(splits)
    output_resolved = output_path.resolve()
    for supplied in supplied_paths:
        path = supplied.expanduser().resolve()
        if path.is_dir():
            for split in wanted_splits:
                candidate = path / f"{split}.jsonl"
                if not candidate.is_file():
                    raise ExtractionError(
                        f"Expected split file does not exist: {candidate}"
                    )
                files.append(candidate)
        elif path.is_file():
            if path.suffix.casefold() != ".jsonl":
                raise ExtractionError(f"Input must be a .jsonl file: {path}")
            files.append(path)
        else:
            raise ExtractionError(f"Input path does not exist: {path}")

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in files:
        resolved = path.resolve()
        if resolved == output_resolved:
            raise ExtractionError("The output JSONL cannot also be an input file")
        if resolved not in seen:
            unique.append(resolved)
            seen.add(resolved)
    if not unique:
        raise ExtractionError("No input JSONL files were found")
    return unique


def count_nonempty_lines(paths: Sequence[Path], tqdm_factory: Any) -> int:
    progress = (
        tqdm_factory(desc="Counting input rows", unit=" rows", dynamic_ncols=True)
        if tqdm_factory is not None
        else None
    )
    count = 0
    try:
        for path in paths:
            with path.open("rb") as handle:
                for raw in handle:
                    if raw.strip():
                        count += 1
                        if progress is not None:
                            progress.update(1)
    finally:
        if progress is not None:
            progress.close()
    return count


def normalized_entities(row: Mapping[str, Any], text: str) -> list[dict[str, Any]]:
    entities: list[dict[str, Any]] = []
    for index, raw_entity in enumerate(as_list(row.get("entities"))):
        entity = as_mapping(raw_entity)
        label = clean_text(entity.get("label"))
        entity_text = entity.get("text")
        start = entity.get("start")
        end = entity.get("end")
        if not label:
            raise ExtractionError(f"Entity {index} has no label")
        if not isinstance(entity_text, str) or not entity_text:
            raise ExtractionError(f"Entity {index} has no text")
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or not (0 <= start < end <= len(text))
        ):
            raise ExtractionError(f"Entity {index} has invalid character offsets")
        actual = text[start:end]
        if actual != entity_text:
            raise ExtractionError(
                f"Entity {index} text mismatch: stored={entity_text!r}, actual={actual!r}"
            )
        normalized = {"start": start, "end": end, "label": label, "text": actual}
        component_kind = clean_text(entity.get("component_kind"))
        if component_kind:
            normalized["component_kind"] = component_kind
        entities.append(normalized)
    entities.sort(key=lambda item: (item["start"], item["end"], item["label"]))
    for previous, current in zip(entities, entities[1:]):
        if current["start"] < previous["end"]:
            raise ExtractionError("Entity spans overlap")
    return entities

def unique_join(values: Iterable[str], separator: str) -> str:
    result: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = clean_text(raw)
        marker = value.casefold()
        if value and marker not in seen:
            result.append(value)
            seen.add(marker)
    return separator.join(result)


def values_for_labels(
    entities: Sequence[Mapping[str, Any]],
    labels: set[str],
    separator: str,
) -> str:
    return unique_join(
        (str(entity["text"]) for entity in entities if entity["label"] in labels),
        separator,
    )


def trim_address_phrase(value: str) -> str:
    return value.strip(" \t\r\n,，、;/／.·-")


def street_phrases(
    text: str,
    entities: Sequence[Mapping[str, Any]],
    include_number: bool,
) -> list[str]:
    """Extract each street and only its immediately adjacent building number."""

    phrases: list[str] = []
    for index, entity in enumerate(entities):
        if entity["label"] != "STREET_NAME":
            continue
        selected = [entity]
        if include_number:
            for neighbour_index in (index - 1, index + 1):
                if not (0 <= neighbour_index < len(entities)):
                    continue
                neighbour = entities[neighbour_index]
                if neighbour["label"] != "BUILDING_NUMBER":
                    continue
                street_kind = clean_text(entity.get("component_kind"))
                number_kind = clean_text(neighbour.get("component_kind"))
                if street_kind and number_kind and (
                    street_kind != "street" or number_kind != "street"
                ):
                    continue
                left, right = sorted(
                    (entity, neighbour), key=lambda item: int(item["start"])
                )
                gap = text[int(left["end"]) : int(right["start"])]
                if SEPARATOR_ONLY.fullmatch(gap):
                    selected.append(neighbour)
        start = min(int(item["start"]) for item in selected)
        end = max(int(item["end"]) for item in selected)
        phrase = trim_address_phrase(text[start:end])
        if phrase:
            phrases.append(phrase)
    return phrases


def observed_fallback(row: Mapping[str, Any], label: str) -> list[str]:
    observed = as_mapping(row.get("observed_components"))
    return [
        clean_text(item) for item in as_list(observed.get(label)) if clean_text(item)
    ]


def fallback_street(
    row: Mapping[str, Any], include_number: bool, separator: str
) -> str:
    names = observed_fallback(row, "STREET_NAME")
    if not names:
        return ""
    if not include_number:
        return unique_join(names, separator)
    numbers = observed_fallback(row, "BUILDING_NUMBER")
    # Without spans, a building number cannot be safely distinguished from a
    # village number when several are present. Only use one unambiguous number.
    number = numbers[0] if len(numbers) == 1 else ""
    language = clean_text(row.get("language"))
    phrases = []
    for name in names:
        if not number:
            phrases.append(name)
        elif language == "en":
            phrases.append(f"{number} {name}")
        else:
            phrases.append(f"{name}{number}")
    return unique_join(phrases, separator)


def normalize_region(value: str, region_format: str) -> str:
    if not value or region_format == "text":
        return value
    normalized = re.sub(r"[.\s]+", " ", value).strip().upper()
    aliases = {
        "H K": "HK",
        "H K ISLAND": "HONG KONG ISLAND",
        "HK ISLAND": "HONG KONG ISLAND",
        "HKI": "HK",
        "KLN": "KLN",
        "N T": "NT",
    }
    normalized = aliases.get(normalized, normalized)
    return REGION_CODES.get(normalized, value)


def normalized_place(value: str) -> str:
    normalized = clean_text(value).casefold()
    normalized = re.sub(r"\s+district$", "", normalized).strip()
    return normalized.removesuffix("區").removesuffix("区").strip()


def choose_district(
    entities: Sequence[Mapping[str, Any]],
    row: Mapping[str, Any],
    district_source: str,
    separator: str,
) -> str:
    sub_district = values_for_labels(entities, {"SUB_DISTRICT"}, separator)
    official = values_for_labels(entities, {"DISTRICT"}, separator)
    if not entities:
        sub_district = unique_join(observed_fallback(row, "SUB_DISTRICT"), separator)
        official = unique_join(observed_fallback(row, "DISTRICT"), separator)
    if district_source == "official":
        return official
    if district_source == "sub_district-then-official":
        return sub_district or official
    return sub_district


def extract_fields(
    row: Mapping[str, Any],
    *,
    district_source: str,
    region_format: str,
    include_street_number: bool,
    multi_value_separator: str,
    keep_duplicate_sub_district: bool,
) -> tuple[str, dict[str, str]]:
    text = row.get("text")
    if not isinstance(text, str):
        raise ExtractionError("Row has no string `text` field")
    entities = normalized_entities(row, text)

    if entities:
        flat = values_for_labels(entities, {"UNIT"}, multi_value_separator)
        floor = values_for_labels(entities, {"FLOOR"}, multi_value_separator)
        block = values_for_labels(entities, {"BLOCK"}, multi_value_separator)
        building_name = values_for_labels(
            entities, {"BUILDING_NAME"}, multi_value_separator
        )
        phase = values_for_labels(entities, {"PHASE"}, multi_value_separator)
        estate_name = values_for_labels(
            entities, {"ESTATE_NAME"}, multi_value_separator
        )
        building_number = values_for_labels(entities, {"BUILDING_NUMBER"}, multi_value_separator)
        village_name = values_for_labels(entities, {"VILLAGE_NAME"}, multi_value_separator)
        street_name = unique_join(
            street_phrases(text, entities, include_street_number),
            multi_value_separator,
        )
        sub_district = values_for_labels(entities, {"SUB_DISTRICT"}, multi_value_separator)
        region = values_for_labels(entities, {"REGION"}, multi_value_separator)
    else:
        flat = unique_join(observed_fallback(row, "UNIT"), multi_value_separator)
        floor = unique_join(observed_fallback(row, "FLOOR"), multi_value_separator)
        block = unique_join(observed_fallback(row, "BLOCK"), multi_value_separator)
        building_name = unique_join(
            observed_fallback(row, "BUILDING_NAME"), multi_value_separator
        )
        phase = unique_join(observed_fallback(row, "PHASE"), multi_value_separator)
        estate_name = unique_join(
            observed_fallback(row, "ESTATE_NAME"), multi_value_separator
        )
        building_number = unique_join(observed_fallback(row, "BUILDING_NUMBER"), multi_value_separator)
        village_name = unique_join(observed_fallback(row, "VILLAGE_NAME"), multi_value_separator)
        street_name = fallback_street(
            row, include_street_number, multi_value_separator
        )
        sub_district = unique_join(observed_fallback(row, "SUB_DISTRICT"), multi_value_separator)
        region = unique_join(observed_fallback(row, "REGION"), multi_value_separator)

    district = choose_district(entities, row, district_source, multi_value_separator)
    if (
        not keep_duplicate_sub_district
        and sub_district
        and district
        and normalized_place(sub_district) == normalized_place(district)
    ):
        sub_district = ""
    values = {
        "flat": flat,
        "floor": floor,
        "block": block,
        "building_name": building_name,
        "phase": phase,
        "estate_name": estate_name,
        "building_number": building_number,
        "street_name": street_name,
        "village_name": village_name,
        "sub_district": sub_district,
        "district": district,
        "region": normalize_region(region, region_format),
    }
    return text, values


def format_output(values: Mapping[str, str], layout: str) -> dict[str, Any]:
    if layout == "flat":
        return {field: values.get(field, "") for field in OUTPUT_FIELDS}
    return {
        "line1": {
            "flat": values.get("flat", ""),
            "floor": values.get("floor", ""),
            "block": values.get("block", ""),
            "building_name": values.get("building_name", ""),
            "phase": values.get("phase", ""),
            "estate_name": values.get("estate_name", ""),
        },
        "line2": {
            "building_number": values.get("building_number", ""),
            "street_name": values.get("street_name", ""),
            "village_name": values.get("village_name", ""),
            "sub_district": values.get("sub_district", ""),
            "district": values.get("district", ""),
            "region": values.get("region", ""),
        },
    }


def write_json_line(handle: TextIO, value: Mapping[str, Any]) -> None:
    handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


def audit_paths(output_path: Path) -> tuple[Path, Path]:
    base = output_path.with_suffix("")
    return Path(f"{base}.audit.json"), Path(f"{base}.audit.md")


def validate_outputs(output_path: Path, overwrite: bool) -> None:
    if output_path.suffix.casefold() != ".jsonl":
        raise ExtractionError("--output must end in .jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_json, audit_md = audit_paths(output_path)
    existing = [path for path in (output_path, audit_json, audit_md) if path.exists()]
    if existing and not overwrite:
        names = ", ".join(str(path) for path in existing)
        raise ExtractionError(f"Output already exists: {names}. Use --overwrite.")


def write_audits(
    output_path: Path,
    input_paths: Sequence[Path],
    counts: Counter[str],
    by_language: Counter[str],
    district_source: str,
    region_format: str,
    include_street_number: bool,
    keep_duplicate_sub_district: bool,
    layout: str,
) -> None:
    audit_json, audit_md = audit_paths(output_path)
    rows_written = counts["rows_written"]
    summary = {
        "input_files": [str(path) for path in input_paths],
        "output_file": str(output_path),
        "configuration": {
            "district_source": district_source,
            "region_format": region_format,
            "include_street_number": include_street_number,
            "keep_duplicate_sub_district": keep_duplicate_sub_district,
            "output_layout": layout,
        },
        "counts": dict(counts),
        "by_language": dict(sorted(by_language.items())),
        "field_presence": {
            field: {
                "nonempty": counts[f"field_{field}_nonempty"],
                "empty": rows_written - counts[f"field_{field}_nonempty"],
                "nonempty_rate": (
                    counts[f"field_{field}_nonempty"] / rows_written
                    if rows_written
                    else 0.0
                ),
            }
            for field in OUTPUT_FIELDS
        },
    }
    audit_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    field_lines = [
        "| Output field | Non-empty | Empty | Non-empty share |",
        "| --- | ---: | ---: | ---: |",
    ]
    for field in OUTPUT_FIELDS:
        nonempty = counts[f"field_{field}_nonempty"]
        empty = rows_written - nonempty
        rate = nonempty / rows_written if rows_written else 0.0
        field_lines.append(f"| {field} | {nonempty:,} | {empty:,} | {rate:.2%} |")

    language_lines = ["| Language | Rows |", "| --- | ---: |"]
    for language, count in by_language.most_common():
        safe = language.replace("|", "\\|")
        language_lines.append(f"| {safe} | {count:,} |")
    if not by_language:
        language_lines.append("| (missing) | 0 |")

    # Assign the joined strings to variables first to avoid the backslash in the f-string
    field_summary = "\n".join(field_lines)
    language_summary = "\n".join(language_lines)

    audit_md.write_text(
        f"""# Compact HK address extraction audit

    - Source rows read: {counts["rows_read"]:,}
    - Compact rows written: {rows_written:,}
    - Invalid rows skipped: {counts["invalid_rows_skipped"]:,}
    - Output layout: `{layout}`
    - District mapping: `{district_source}`
    - Region format: `{region_format}`
    - Street number included: `{include_street_number}`
    - Duplicate sub-district retained: `{keep_duplicate_sub_district}`

    ## Output-field presence

    {field_summary}

    ## Source language counts

    {language_summary}

    Only visible entity spans or `observed_components` are used. Canonical
    `components` metadata is deliberately ignored, so a component omitted from the
    input cannot leak into the target.
    """,
        encoding="utf-8",
    )


def extract_dataset(args: argparse.Namespace) -> Path:
    output_path = args.output.expanduser().resolve()
    validate_outputs(output_path, args.overwrite)
    input_paths = discover_inputs(args.input, args.splits, output_path)
    tqdm_factory = load_tqdm(args.progress)
    total_rows = count_nonempty_lines(input_paths, tqdm_factory)
    LOGGER.info("Found %s non-empty input rows", f"{total_rows:,}")
    progress = (
        tqdm_factory(
            total=total_rows,
            desc="Extracting compact rows",
            unit=" rows",
            dynamic_ncols=True,
        )
        if tqdm_factory is not None
        else None
    )
    counts: Counter[str] = Counter()
    by_language: Counter[str] = Counter()

    try:
        with output_path.open("w", encoding="utf-8") as output:
            for path in input_paths:
                with path.open("r", encoding="utf-8") as source:
                    for line_number, line in enumerate(source, start=1):
                        if not line.strip():
                            continue
                        counts["rows_read"] += 1
                        if progress is not None:
                            progress.update(1)
                        try:
                            row = json.loads(line)
                            if not isinstance(row, Mapping):
                                raise ExtractionError("JSON value is not an object")
                            text, values = extract_fields(
                                row,
                                district_source=args.district_source,
                                region_format=args.region_format,
                                include_street_number=args.include_street_number,
                                multi_value_separator=args.multi_value_separator,
                                keep_duplicate_sub_district=args.keep_duplicate_sub_district,
                            )
                        except (json.JSONDecodeError, ExtractionError) as exc:
                            if not args.skip_invalid:
                                raise ExtractionError(
                                    f"Invalid row at {path}:{line_number}: {exc}"
                                ) from exc
                            counts["invalid_rows_skipped"] += 1
                            LOGGER.warning("Skipping %s:%s: %s", path, line_number, exc)
                            continue

                        compact = {
                            "input": text,
                            "output": format_output(values, args.output_layout),
                        }
                        write_json_line(output, compact)
                        counts["rows_written"] += 1
                        language = clean_text(row.get("language")) or "<missing>"
                        by_language[language] += 1
                        for field, value in values.items():
                            if value:
                                counts[f"field_{field}_nonempty"] += 1
    finally:
        if progress is not None:
            progress.close()

    write_audits(
        output_path,
        input_paths,
        counts,
        by_language,
        args.district_source,
        args.region_format,
        args.include_street_number,
        args.keep_duplicate_sub_district,
        args.output_layout,
    )
    LOGGER.info(
        "Wrote %s compact rows to %s", f"{counts['rows_written']:,}", output_path
    )
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Keep only model-useful HK address fields and write compact "
            "input/output JSONL rows."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        required=True,
        help="One or more generated JSONL files or dataset directories",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "validation", "test", "all"),
        default=("train",),
        help="Files read when an input is a dataset directory",
    )
    parser.add_argument(
        "--output-layout",
        choices=("nested", "flat"),
        default="nested",
        help="Nest fields under line1/line2 or keep one flat output object",
    )
    parser.add_argument(
        "--district-source",
        choices=("sub_district", "official", "sub_district-then-official"),
        default="official",
        help=(
            "Map district from LOCATION (for example Lam Tin), official DISTRICT "
            "(for example Kwun Tong District), or sub_district with official fallback"
        ),
    )
    parser.add_argument(
        "--region-format",
        choices=("code", "text"),
        default="code",
        help="Normalize visible region values to HK/KLN/NT or preserve input text",
    )
    parser.add_argument(
        "--include-street-number",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include an adjacent building number in street_name",
    )
    parser.add_argument(
        "--keep-duplicate-sub-district",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Keep sub_district when it normalizes to the selected district; "
            "the default removes the duplicate deterministically"
        ),
    )
    parser.add_argument(
        "--multi-value-separator",
        default=" / ",
        help="Separator if more than one visible value maps to one output field",
    )
    parser.add_argument(
        "--skip-invalid",
        action="store_true",
        help="Skip malformed rows instead of stopping",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show tqdm counting and extraction progress",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    if not args.multi_value_separator:
        parser.exit(2, "error: --multi-value-separator cannot be empty\n")
    try:
        output = extract_dataset(args)
    except (ExtractionError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
