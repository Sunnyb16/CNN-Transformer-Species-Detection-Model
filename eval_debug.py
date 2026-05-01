import json
from pathlib import Path

import numpy as np
import torch


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()

    all_probs = []
    all_targets = []
    all_sample_infos = []

    dataset = getattr(loader, "dataset", None)

    sample_index = 0
    for inputs, targets in loader:
        batch_size = inputs.size(0)
        inputs = inputs.to(device, non_blocking=True)

        logits = model(inputs)
        probs = torch.sigmoid(logits).cpu().numpy()
        targets_np = targets.numpy()

        all_probs.append(probs)
        all_targets.append(targets_np)

        for local_idx in range(batch_size):
            info = {}
            if dataset is not None and hasattr(dataset, "samples"):
                raw_sample = dataset.samples[sample_index + local_idx]
                info = {
                    "spec_path": raw_sample.get("spec_path"),
                    "source_path": raw_sample.get("source_path"),
                    "source_interval": raw_sample.get("source_interval"),
                    "phase_group": raw_sample.get("phase_group"),
                    "labels": raw_sample.get("labels"),
                    "file_id": raw_sample.get("file_id"),
                }
            all_sample_infos.append(info)

        sample_index += batch_size

    if not all_probs:
        raise RuntimeError("No predictions collected from loader.")

    return {
        "probs": np.concatenate(all_probs, axis=0),
        "targets": np.concatenate(all_targets, axis=0),
        "sample_infos": all_sample_infos,
    }


def _top_labels(probs_row, class_list, limit=5):
    top_indices = np.argsort(probs_row)[::-1][:limit]
    return [
        {
            "label": class_list[idx],
            "score": float(probs_row[idx]),
        }
        for idx in top_indices
    ]


def build_error_report(
    probs,
    targets,
    sample_infos,
    class_list,
    decision_threshold=0.5,
    low_conf_threshold=0.6,
    top_k=5,
):
    false_positives = []
    false_negatives = []
    low_confidence = []
    class_summary = []

    pred_binary = (probs >= decision_threshold).astype(np.int32)
    true_binary = targets.astype(np.int32)

    for row_idx in range(len(probs)):
        info = sample_infos[row_idx] if row_idx < len(sample_infos) else {}
        probs_row = probs[row_idx]
        pred_row = pred_binary[row_idx]
        true_row = true_binary[row_idx]

        fp_indices = np.where((pred_row == 1) & (true_row == 0))[0]
        fn_indices = np.where((pred_row == 0) & (true_row == 1))[0]

        if len(fp_indices) > 0:
            false_positives.append(
                {
                    "sample": info,
                    "false_positive_labels": [
                        {
                            "label": class_list[idx],
                            "score": float(probs_row[idx]),
                        }
                        for idx in fp_indices
                    ],
                    "top_predictions": _top_labels(probs_row, class_list, limit=top_k),
                }
            )

        if len(fn_indices) > 0:
            false_negatives.append(
                {
                    "sample": info,
                    "false_negative_labels": [
                        {
                            "label": class_list[idx],
                            "score": float(probs_row[idx]),
                        }
                        for idx in fn_indices
                    ],
                    "top_predictions": _top_labels(probs_row, class_list, limit=top_k),
                }
            )

        low_conf_mask = probs_row < low_conf_threshold
        predicted_positive_mask = pred_row == 1
        true_positive_mask = true_row == 1
        flagged_indices = np.where(low_conf_mask & (predicted_positive_mask | true_positive_mask))[0]

        if len(flagged_indices) > 0:
            low_confidence.append(
                {
                    "sample": info,
                    "labels_under_threshold": [
                        {
                            "label": class_list[idx],
                            "score": float(probs_row[idx]),
                            "is_true_label": bool(true_row[idx]),
                            "is_predicted_positive": bool(pred_row[idx]),
                        }
                        for idx in flagged_indices
                    ],
                    "top_predictions": _top_labels(probs_row, class_list, limit=top_k),
                }
            )

    for idx, label in enumerate(class_list):
        true_mask = true_binary[:, idx] == 1
        pred_mask = pred_binary[:, idx] == 1
        fp_mask = pred_mask & ~true_mask
        fn_mask = ~pred_mask & true_mask
        low_conf_mask = (
            (probs[:, idx] < low_conf_threshold)
            & (true_mask | pred_mask)
        )

        true_scores = probs[true_mask, idx]
        pred_scores = probs[pred_mask, idx]

        class_summary.append(
            {
                "label": label,
                "true_count": int(np.sum(true_mask)),
                "predicted_positive_count": int(np.sum(pred_mask)),
                "false_positive_count": int(np.sum(fp_mask)),
                "false_negative_count": int(np.sum(fn_mask)),
                "low_confidence_count": int(np.sum(low_conf_mask)),
                "avg_score_when_true": (
                    float(np.mean(true_scores)) if true_scores.size > 0 else None
                ),
                "avg_score_when_predicted_positive": (
                    float(np.mean(pred_scores)) if pred_scores.size > 0 else None
                ),
            }
        )

    class_summary = sorted(
        class_summary,
        key=lambda item: (
            item["false_negative_count"],
            item["false_positive_count"],
            item["low_confidence_count"],
            item["true_count"],
        ),
        reverse=True,
    )

    report = {
        "summary": {
            "num_samples": int(len(probs)),
            "decision_threshold": decision_threshold,
            "low_conf_threshold": low_conf_threshold,
            "samples_with_false_positives": int(len(false_positives)),
            "samples_with_false_negatives": int(len(false_negatives)),
            "samples_with_low_confidence_labels": int(len(low_confidence)),
        },
        "class_summary": class_summary,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "low_confidence": low_confidence,
    }
    return report


def save_error_report(report, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return output_path
