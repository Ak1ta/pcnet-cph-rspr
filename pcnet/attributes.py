from __future__ import annotations

import csv
import json
import re
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple
from xml.etree import ElementTree

from .config import ATTRIBUTE_NAMES


_CELL_REFERENCE = re.compile(r"([A-Z]+)([0-9]+)")


def canonical_name(value: str) -> str:
    return Path(str(value).strip()).stem.casefold()


def _column_index(cell_reference: str) -> int:
    match = _CELL_REFERENCE.fullmatch(cell_reference)
    if match is None:
        raise ValueError("Invalid Excel cell reference: {}".format(cell_reference))
    result = 0
    for character in match.group(1):
        result = result * 26 + ord(character) - ord("A") + 1
    return result - 1


def _read_shared_strings(archive: zipfile.ZipFile) -> List[str]:
    try:
        payload = archive.read("xl/sharedStrings.xml")
    except KeyError:
        return []
    root = ElementTree.fromstring(payload)
    namespace = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    strings = []
    for item in root.findall(namespace + "si"):
        strings.append("".join(node.text or "" for node in item.iter(namespace + "t")))
    return strings


def _read_first_sheet_rows(path: Path) -> List[List[str]]:
    with zipfile.ZipFile(str(path)) as archive:
        shared_strings = _read_shared_strings(archive)
        worksheet = ElementTree.fromstring(archive.read("xl/worksheets/sheet1.xml"))

    namespace = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    rows: List[List[str]] = []
    for row in worksheet.iter(namespace + "row"):
        values: Dict[int, str] = {}
        for cell in row.findall(namespace + "c"):
            reference = cell.attrib.get("r")
            if not reference:
                continue
            index = _column_index(reference)
            cell_type = cell.attrib.get("t")
            value_node = cell.find(namespace + "v")
            if cell_type == "inlineStr":
                inline = cell.find(namespace + "is")
                value = "" if inline is None else "".join(
                    node.text or "" for node in inline.iter(namespace + "t")
                )
            elif value_node is None:
                value = ""
            elif cell_type == "s":
                value = shared_strings[int(value_node.text or "0")]
            else:
                value = value_node.text or ""
            values[index] = value
        if values:
            width = max(values) + 1
            rows.append([values.get(index, "") for index in range(width)])
    return rows


def _read_csv_rows(path: Path) -> List[List[str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return [list(row) for row in csv.reader(handle)]


def load_attribute_table(path: str) -> Tuple[Dict[str, Tuple[float, ...]], Tuple[str, ...]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError("Attribute table does not exist: {}".format(source))
    suffix = source.suffix.casefold()
    if suffix == ".xlsx":
        rows = _read_first_sheet_rows(source)
    elif suffix == ".csv":
        rows = _read_csv_rows(source)
    else:
        raise ValueError("Attribute table must be .xlsx or .csv: {}".format(source))
    if not rows:
        raise ValueError("Attribute table is empty: {}".format(source))

    header = tuple(str(value).strip() for value in rows[0])
    expected = ("name",) + ATTRIBUTE_NAMES
    if header != expected:
        raise ValueError(
            "Unexpected attribute columns in {}. Expected {}, got {}.".format(
                source, expected, header
            )
        )

    mapping: Dict[str, Tuple[float, ...]] = {}
    for excel_row, row in enumerate(rows[1:], start=2):
        if not row or not str(row[0]).strip():
            continue
        if len(row) != len(expected):
            raise ValueError(
                "Row {} in {} has {} columns; expected {}.".format(
                    excel_row, source, len(row), len(expected)
                )
            )
        key = canonical_name(row[0])
        if key in mapping:
            raise ValueError("Duplicate sample '{}' in {}.".format(row[0], source))
        labels = []
        for name, raw_value in zip(ATTRIBUTE_NAMES, row[1:]):
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                raise ValueError(
                    "Invalid {} label '{}' at row {} in {}.".format(
                        name, raw_value, excel_row, source
                    )
                )
            if value not in {0.0, 1.0}:
                raise ValueError(
                    "{} label must be 0 or 1 at row {} in {}, got {}.".format(
                        name, excel_row, source, value
                    )
                )
            labels.append(value)
        mapping[key] = tuple(labels)
    return mapping, header


def labels_for_name(
    mapping: Mapping[str, Sequence[float]], sample_name: str
):
    import torch

    key = canonical_name(sample_name)
    if key not in mapping:
        raise KeyError("No challenge labels found for sample: {}".format(sample_name))
    return torch.tensor(mapping[key], dtype=torch.float32)


def validate_sample_coverage(
    mapping: Mapping[str, Sequence[float]], sample_names: Iterable[str]
) -> None:
    expected = {canonical_name(name) for name in sample_names}
    actual = set(mapping)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        fragments = []
        if missing:
            fragments.append("missing {} (examples: {})".format(len(missing), missing[:5]))
        if extra:
            fragments.append("extra {} (examples: {})".format(len(extra), extra[:5]))
        raise ValueError("Attribute/image coverage mismatch: " + "; ".join(fragments))


def attribute_statistics(mapping: Mapping[str, Sequence[float]]) -> Dict[str, object]:
    counts = [0] * len(ATTRIBUTE_NAMES)
    cardinalities: Dict[int, int] = {}
    for labels in mapping.values():
        active = int(sum(labels))
        cardinalities[active] = cardinalities.get(active, 0) + 1
        for index, value in enumerate(labels):
            counts[index] += int(value)
    total = len(mapping)
    return {
        "samples": total,
        "mean_attributes_per_sample": sum(
            cardinality * count for cardinality, count in cardinalities.items()
        ) / max(total, 1),
        "positive_counts": dict(zip(ATTRIBUTE_NAMES, counts)),
        "positive_rates": {
            name: count / max(total, 1) for name, count in zip(ATTRIBUTE_NAMES, counts)
        },
        "cardinality_counts": dict(sorted(cardinalities.items())),
    }


def export_csv(mapping: Mapping[str, Sequence[float]], output_path: str) -> None:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("name",) + ATTRIBUTE_NAMES)
        for name in sorted(mapping):
            writer.writerow((name,) + tuple(int(value) for value in mapping[name]))


def save_statistics(statistics: Mapping[str, object], output_path: str) -> None:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(statistics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
