import ast
import json
import pickle
from collections import defaultdict
from pathlib import Path

import pandas as pd

from src.paths import resolve_path


def _parse_listish(value):
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            return [value]
    return []


def _normalize_source_path(value):
    if value is None:
        return None
    path = Path(value)
    return str(path)


def _group_samples_by_source(samples):
    grouped = defaultdict(list)
    for sample in samples:
        grouped[_normalize_source_path(sample.get("source_path"))].append(sample)
    return grouped


def inspect_debug_outputs(debug_dir, base_dir="."):
    debug_path = Path(resolve_path(debug_dir, base_dir))
    rows_path = debug_path / "debug_rows.csv"
    samples_path = debug_path / "debug_samples.pkl"

    rows_df = pd.read_csv(rows_path)
    with open(samples_path, "rb") as handle:
        samples = pickle.load(handle)

    grouped_samples = _group_samples_by_source(samples)

    records = []
    for _, row in rows_df.iterrows():
        source_path = _normalize_source_path(resolve_path(row["full_path"], base_dir))
        row_samples = grouped_samples.get(source_path, [])
        if not row_samples:
            rel_source_path = _normalize_source_path(row["full_path"])
            row_samples = grouped_samples.get(rel_source_path, [])
        sample_intervals = [
            sample.get("source_interval")
            for sample in row_samples
            if sample.get("source_interval") is not None
        ]
        generated_labels = []
        fallback_flags = []
        chunk_origins = []
        for sample in row_samples:
            generated_labels.append(sample.get("labels", []))
            fallback_flags.append(bool(sample.get("used_fallback", False)))
            chunk_origins.append(sample.get("chunk_origin", "unknown"))

        records.append(
            {
                "filename": row["filename"],
                "phase_group": row["phase_group"],
                "primary_label": row["primary_label"],
                "secondary_labels": _parse_listish(row.get("secondary_labels")),
                "all_labels": _parse_listish(row.get("all_labels")),
                "start_time": row.get("start_time"),
                "end_time": row.get("end_time"),
                "generated_chunk_count": len(row_samples),
                "generated_labels_per_chunk": generated_labels,
                "generated_intervals": sample_intervals,
                "chunk_origin_per_chunk": chunk_origins,
                "chunk_origin_set": sorted(set(chunk_origins)),
                "used_fallback_per_chunk": fallback_flags,
                "used_fallback_any": any(fallback_flags),
                "source_path": source_path,
            }
        )

    summary = {
        "debug_dir": str(debug_path),
        "selected_rows": len(records),
        "generated_samples": len(samples),
        "rows_with_zero_chunks": sum(1 for record in records if record["generated_chunk_count"] == 0),
        "rows_with_multiple_chunks": sum(1 for record in records if record["generated_chunk_count"] > 1),
        "rows_with_any_fallback_chunks": sum(1 for record in records if record["used_fallback_any"]),
        "generated_fallback_chunks": sum(
            sum(1 for flag in record["used_fallback_per_chunk"] if flag)
            for record in records
        ),
        "generated_chunks_by_origin": (
            pd.Series(
                [
                    origin
                    for record in records
                    for origin in record["chunk_origin_per_chunk"]
                ],
                dtype="object",
            )
            .value_counts()
            .to_dict()
        ),
    }

    return summary, records


def write_debug_inspection(debug_dir, output_path=None, base_dir="."):
    summary, records = inspect_debug_outputs(debug_dir, base_dir=base_dir)
    debug_path = Path(resolve_path(debug_dir, base_dir))

    if output_path is None:
        output_path = debug_path / "inspection.json"
    else:
        output_path = Path(output_path)

    payload = {
        "summary": summary,
        "records": records,
    }
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    return payload, output_path
