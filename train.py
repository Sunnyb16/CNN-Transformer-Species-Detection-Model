import argparse
import copy
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import BirdChunkDataset
from src.eval_debug import build_error_report, collect_predictions, save_error_report
from src.model import BirdResNet
from src.paths import resolve_path


DEFAULT_CURRICULUM = [
    ("clean",),
    ("clean", "semi"),
    ("clean", "semi", "soundscape"),
    ("clean", "semi", "soundscape", "noisy"),
]

DEFAULT_AUG_PARAMS = {
    "pitch_prob": 0.4,
    "pitch_range": (-1.5, 1.5),
    "stretch_prob": 0.4,
    "stretch_range": (0.9, 1.1),
    "shift_prob": 0.5,
    "shift_max_sec": 1.0,
    "noise_prob": 0.5,
    "noise_std": 0.005,
    "time_mask_prob": 0.5,
    "time_mask_param": 12,
    "num_time_masks": 1,
    "freq_mask_prob": 0.5,
    "freq_mask_param": 12,
    "num_freq_masks": 1,
}


def load_fold_samples(base_output_dir, fold, base_dir="."):
    fold_dir = Path(resolve_path(base_output_dir, base_dir)) / f"fold_{fold}"

    with open(fold_dir / "train_samples.pkl", "rb") as handle:
        train_samples = pickle.load(handle)
    with open(fold_dir / "val_samples.pkl", "rb") as handle:
        val_samples = pickle.load(handle)

    return train_samples, val_samples


def build_dataloaders(
    train_samples,
    val_samples,
    label_to_idx,
    batch_size=32,
    num_workers=0,
    augment=False,
    aug_params=None,
    base_dir=".",
    train_secondary_label_total_weight=None,
):
    train_ds = BirdChunkDataset(
        train_samples,
        augment=augment,
        aug_params=aug_params,
        base_dir=base_dir,
        label_to_idx=label_to_idx,
        secondary_label_total_weight=train_secondary_label_total_weight,
    )
    val_ds = BirdChunkDataset(
        val_samples,
        augment=False,
        base_dir=base_dir,
        label_to_idx=label_to_idx,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def filter_samples_by_phases(samples, phases):
    phase_set = set(phases)
    filtered = [
        sample
        for sample in samples
        if sample.get("phase_group") in phase_set
    ]
    if filtered:
        return filtered
    return samples


def curriculum_phases_for_epoch(epoch, total_epochs, curriculum=None):
    curriculum = curriculum or DEFAULT_CURRICULUM

    if total_epochs <= 0:
        return curriculum[-1]

    boundaries = np.linspace(1, total_epochs + 1, num=len(curriculum) + 1)
    for idx in range(len(curriculum)):
        start = int(np.floor(boundaries[idx]))
        end = int(np.floor(boundaries[idx + 1]))
        if idx == len(curriculum) - 1:
            end = total_epochs + 1
        if start <= epoch < end:
            return curriculum[idx]

    return curriculum[-1]


def normalize_curriculum(curriculum):
    if curriculum is None:
        return [tuple(stage) for stage in DEFAULT_CURRICULUM]

    normalized = []
    for stage in curriculum:
        if isinstance(stage, str):
            labels = [label.strip() for label in stage.split(",") if label.strip()]
        else:
            labels = [str(label).strip() for label in stage if str(label).strip()]

        if not labels:
            continue
        normalized.append(tuple(labels))

    if not normalized:
        raise ValueError("Curriculum must contain at least one non-empty stage.")

    return normalized


def parse_curriculum_arg(value):
    if not value:
        return None

    path = Path(value)
    if path.exists():
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return normalize_curriculum(payload)

    stages = []
    for raw_stage in value.split("|"):
        labels = [label.strip() for label in raw_stage.split(",") if label.strip()]
        if labels:
            stages.append(labels)

    return normalize_curriculum(stages)


def normalize_aug_params(aug_params):
    if aug_params is None:
        return dict(DEFAULT_AUG_PARAMS)

    normalized = dict(DEFAULT_AUG_PARAMS)
    normalized.update(aug_params)

    for key in ("pitch_range", "stretch_range"):
        value = normalized.get(key)
        if isinstance(value, list):
            normalized[key] = tuple(value)

    return normalized


def parse_aug_config_arg(value):
    if not value:
        return None

    path = Path(value)
    if path.exists():
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        payload = json.loads(value)

    if not isinstance(payload, dict):
        raise ValueError("Augmentation config must be a JSON object.")

    return normalize_aug_params(payload)


def compute_pos_weight(samples):
    targets = np.stack([sample["target"] for sample in samples])
    pos_counts = targets.sum(axis=0)
    neg_counts = len(targets) - pos_counts
    pos_weight = neg_counts / (pos_counts + 1e-6)
    pos_weight = np.clip(pos_weight, 1.0, 20.0)
    return torch.tensor(pos_weight, dtype=torch.float32)


def compute_macro_auc_from_arrays(targets, probs):
    targets = np.asarray(targets)
    probs = np.asarray(probs)

    if targets.ndim != 2 or probs.ndim != 2:
        raise ValueError("targets and probs must both be 2D arrays.")

    positive_mask = targets.sum(axis=0) > 0
    if not np.any(positive_mask):
        return None

    return float(
        roc_auc_score(
            targets[:, positive_mask],
            probs[:, positive_mask],
            average="macro",
        )
    )


def run_epoch(model, loader, criterion, optimizer, device, epoch=None, total_epochs=None):
    model.train()
    total_loss = 0.0

    desc = "train"
    if epoch is not None and total_epochs is not None:
        desc = f"train {epoch}/{total_epochs}"

    progress = tqdm(loader, desc=desc, leave=False)
    for inputs, targets in progress:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
    epoch=None,
    total_epochs=None,
    species_class_count=None,
):
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_targets = []

    desc = "val"
    if epoch is not None and total_epochs is not None:
        desc = f"val {epoch}/{total_epochs}"

    progress = tqdm(loader, desc=desc, leave=False)
    for inputs, targets in progress:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(inputs)
        loss = criterion(logits, targets)
        total_loss += loss.item() * inputs.size(0)
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_targets.append(targets.cpu().numpy())
        progress.set_postfix(loss=f"{loss.item():.4f}")

    val_loss = total_loss / len(loader.dataset)
    val_auc = None
    if all_probs:
        targets_for_auc = np.concatenate(all_targets, axis=0)
        probs_for_auc = np.concatenate(all_probs, axis=0)
        if species_class_count is not None:
            targets_for_auc = targets_for_auc[:, :species_class_count]
            probs_for_auc = probs_for_auc[:, :species_class_count]
        val_auc = compute_macro_auc_from_arrays(
            targets_for_auc,
            probs_for_auc,
        )

    return val_loss, val_auc


def train_fold(
    fold,
    base_dir=".",
    data_processed_dir="data_processed",
    splits_dir="splits",
    output_dir="checkpoints",
    epochs=10,
    batch_size=32,
    lr=1e-3,
    num_workers=0,
    augment=False,
    curriculum=False,
    curriculum_stages=None,
    aug_params=None,
    debug_errors=False,
    debug_decision_threshold=0.5,
    debug_low_conf_threshold=0.6,
    debug_top_k=5,
    dropout=0.2,
    early_stopping_patience=None,
    secondary_label_total_weight=None,
    device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    splits_path = Path(splits_dir)
    with open(splits_path / "class_list.json", "r", encoding="utf-8") as handle:
        class_list = json.load(handle)
    species_class_list_path = splits_path / "species_class_list.json"
    if species_class_list_path.exists():
        with open(species_class_list_path, "r", encoding="utf-8") as handle:
            species_class_list = json.load(handle)
    else:
        species_class_list = class_list
    species_class_count = len(species_class_list)

    aug_params = normalize_aug_params(aug_params)
    curriculum_stages = normalize_curriculum(curriculum_stages)

    train_samples, val_samples = load_fold_samples(data_processed_dir, fold, base_dir=base_dir)
    _, val_loader = build_dataloaders(
        train_samples=train_samples,
        val_samples=val_samples,
        label_to_idx={label: idx for idx, label in enumerate(class_list)},
        batch_size=batch_size,
        num_workers=num_workers,
        augment=False,
        base_dir=base_dir,
    )

    label_to_idx = {label: idx for idx, label in enumerate(class_list)}

    model = BirdResNet(num_classes=len(class_list), dropout=dropout).to(device)
    pos_weight = compute_pos_weight(train_samples).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(output_dir) / f"fold_{fold}_best.pt"

    best_val_loss = float("inf")
    best_val_auc = float("-inf")
    best_epoch = None
    best_state_dict = None
    history = []
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        active_phases = None
        epoch_train_samples = train_samples
        if curriculum:
            active_phases = curriculum_phases_for_epoch(
                epoch,
                epochs,
                curriculum=curriculum_stages,
            )
            epoch_train_samples = filter_samples_by_phases(train_samples, active_phases)

        train_loader, _ = build_dataloaders(
            train_samples=epoch_train_samples,
            val_samples=val_samples,
            label_to_idx=label_to_idx,
            batch_size=batch_size,
            num_workers=num_workers,
            augment=augment,
            aug_params=aug_params,
            base_dir=base_dir,
            train_secondary_label_total_weight=secondary_label_total_weight,
        )

        train_loss = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch=epoch,
            total_epochs=epochs,
            species_class_count=species_class_count,
        )
        val_loss, val_auc = evaluate(
            model,
            val_loader,
            criterion,
            device,
            epoch=epoch,
            total_epochs=epochs,
        )

        epoch_result = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_macro_auc": val_auc,
            "train_samples": len(epoch_train_samples),
            "active_phases": list(active_phases) if active_phases is not None else None,
        }
        history.append(epoch_result)
        phase_msg = ""
        if active_phases is not None:
            phase_msg = (
                f" phases={','.join(active_phases)}"
                f" train_samples={len(epoch_train_samples)}"
            )
        auc_msg = "n/a" if val_auc is None else f"{val_auc:.5f}"
        print(
            f"fold={fold} epoch={epoch}/{epochs} "
            f"train_loss={train_loss:.5f} val_loss={val_loss:.5f} "
            f"val_macro_auc={auc_msg}{phase_msg}"
        )

        improved = False
        if val_auc is not None:
            improved = val_auc > best_val_auc
        else:
            improved = val_loss < best_val_loss

        if improved:
            best_val_loss = val_loss
            if val_auc is not None:
                best_val_auc = val_auc
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state_dict = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    "fold": fold,
                    "class_list": class_list,
                    "species_class_list": species_class_list,
                    "model_state_dict": model.state_dict(),
                    "history": history,
                    "best_val_loss": best_val_loss,
                    "best_val_auc": None if best_val_auc == float("-inf") else best_val_auc,
                    "best_epoch": best_epoch,
                    "dropout": dropout,
                    "secondary_label_total_weight": secondary_label_total_weight,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        if (
            early_stopping_patience is not None
            and early_stopping_patience >= 0
            and epochs_without_improvement >= early_stopping_patience
        ):
            print(
                f"early stopping triggered at epoch={epoch} "
                f"(best_epoch={best_epoch}, patience={early_stopping_patience})"
            )
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    debug_report_path = None
    if debug_errors:
        prediction_payload = collect_predictions(model, val_loader, device)
        debug_report = build_error_report(
            probs=prediction_payload["probs"][:, :species_class_count],
            targets=prediction_payload["targets"][:, :species_class_count],
            sample_infos=prediction_payload["sample_infos"],
            class_list=species_class_list,
            decision_threshold=debug_decision_threshold,
            low_conf_threshold=debug_low_conf_threshold,
            top_k=debug_top_k,
        )
        debug_report_path = save_error_report(
            debug_report,
            Path(output_dir) / f"fold_{fold}_debug_report.json",
        )
        print(f"saved debug report -> {debug_report_path}")

    return {
        "checkpoint_path": str(checkpoint_path),
        "best_val_loss": best_val_loss,
        "best_val_auc": None if best_val_auc == float("-inf") else best_val_auc,
        "best_epoch": best_epoch,
        "history": history,
        "debug_report_path": str(debug_report_path) if debug_report_path else None,
        "secondary_label_total_weight": secondary_label_total_weight,
    }


def main():
    parser = argparse.ArgumentParser(description="Train one BirdCLEF fold.")
    parser.add_argument("--base-dir", default=".")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--data-processed-dir", default="data_processed")
    parser.add_argument("--splits-dir", default="splits")
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--aug-config", default=None)
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--curriculum-config", default=None)
    parser.add_argument("--debug-errors", action="store_true")
    parser.add_argument("--debug-decision-threshold", type=float, default=0.5)
    parser.add_argument("--debug-low-conf-threshold", type=float, default=0.6)
    parser.add_argument("--debug-top-k", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--secondary-label-total-weight", type=float, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    train_fold(
        fold=args.fold,
        base_dir=args.base_dir,
        data_processed_dir=args.data_processed_dir,
        splits_dir=args.splits_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        augment=args.augment,
        curriculum=args.curriculum,
        curriculum_stages=parse_curriculum_arg(args.curriculum_config),
        aug_params=parse_aug_config_arg(args.aug_config),
        debug_errors=args.debug_errors,
        debug_decision_threshold=args.debug_decision_threshold,
        debug_low_conf_threshold=args.debug_low_conf_threshold,
        debug_top_k=args.debug_top_k,
        dropout=args.dropout,
        early_stopping_patience=args.early_stopping_patience,
        secondary_label_total_weight=args.secondary_label_total_weight,
        device=args.device,
    )


if __name__ == "__main__":
    main()
