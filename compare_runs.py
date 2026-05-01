import json
from pathlib import Path

import torch


def _resolve_run_name(path_value, explicit_name=None):
    if explicit_name:
        return explicit_name
    return Path(path_value).stem


def _load_checkpoint_summary(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    history = checkpoint.get("history", [])
    best_val_loss = checkpoint.get("best_val_loss")

    best_epoch = None
    if history and best_val_loss is not None:
        matches = [item for item in history if item.get("val_loss") == best_val_loss]
        if matches:
            best_epoch = matches[0].get("epoch")

    return {
        "checkpoint_path": str(checkpoint_path),
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "history": history,
    }


def _load_debug_report_summary(report_path):
    with open(report_path, "r", encoding="utf-8") as handle:
        report = json.load(handle)

    summary = report.get("summary", {})
    return {
        "report_path": str(report_path),
        "num_samples": summary.get("num_samples"),
        "samples_with_false_positives": summary.get("samples_with_false_positives"),
        "samples_with_false_negatives": summary.get("samples_with_false_negatives"),
        "samples_with_low_confidence_labels": summary.get("samples_with_low_confidence_labels"),
        "decision_threshold": summary.get("decision_threshold"),
        "low_conf_threshold": summary.get("low_conf_threshold"),
    }


def compare_training_runs(run_specs):
    rows = []

    for spec in run_specs:
        checkpoint_path = Path(spec["checkpoint"])
        debug_report_path = Path(spec["debug_report"])
        run_name = _resolve_run_name(checkpoint_path, spec.get("name"))

        checkpoint_summary = _load_checkpoint_summary(checkpoint_path)
        debug_summary = _load_debug_report_summary(debug_report_path)

        rows.append(
            {
                "run_name": run_name,
                "best_val_loss": checkpoint_summary["best_val_loss"],
                "best_epoch": checkpoint_summary["best_epoch"],
                "samples_with_false_positives": debug_summary["samples_with_false_positives"],
                "samples_with_false_negatives": debug_summary["samples_with_false_negatives"],
                "samples_with_low_confidence_labels": debug_summary["samples_with_low_confidence_labels"],
                "checkpoint_path": checkpoint_summary["checkpoint_path"],
                "debug_report_path": debug_summary["report_path"],
            }
        )

    sorted_rows = sorted(
        rows,
        key=lambda item: (
            float("inf") if item["best_val_loss"] is None else item["best_val_loss"],
            item["samples_with_false_negatives"] if item["samples_with_false_negatives"] is not None else float("inf"),
            item["samples_with_false_positives"] if item["samples_with_false_positives"] is not None else float("inf"),
        ),
    )

    return {
        "runs": rows,
        "ranking": sorted_rows,
        "best_run": sorted_rows[0] if sorted_rows else None,
    }


def write_training_comparison(run_specs, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = compare_training_runs(run_specs)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    return payload, output_path
