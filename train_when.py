import argparse
import copy
import json
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.config import DURATION, FMAX, FMIN, HOP_LENGTH, N_FFT, N_MELS, SR
from src.paths import resolve_path
from src.when_model import WhenNet


class WhenClipDataset(Dataset):
    def __init__(self, rows, base_dir=".", random_crop=True):
        self.rows = list(rows)
        self.base_dir = base_dir
        self.random_crop = random_crop

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        audio_path = resolve_path(row["full_path"], self.base_dir)
        y, _ = librosa.load(audio_path, sr=SR, mono=True)

        target_len = int(round(SR * DURATION))
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        elif len(y) > target_len:
            if self.random_crop:
                start = np.random.randint(0, len(y) - target_len + 1)
            else:
                start = max(0, (len(y) - target_len) // 2)
            y = y[start:start + target_len]

        mel = librosa.feature.melspectrogram(
            y=y,
            sr=SR,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            fmin=FMIN,
            fmax=FMAX,
        )
        log_mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)

        return torch.from_numpy(log_mel).unsqueeze(0), torch.tensor(1.0, dtype=torch.float32)


def mmm_loss(frame_logits, clip_labels):
    probs = torch.sigmoid(frame_logits)
    max_probs = probs.max(dim=1).values
    mean_probs = probs.mean(dim=1)
    min_probs = probs.min(dim=1).values

    loss_max = F.binary_cross_entropy(max_probs, clip_labels)
    loss_mean = F.binary_cross_entropy(mean_probs, clip_labels * 0.5)
    loss_min = F.binary_cross_entropy(min_probs, torch.zeros_like(clip_labels))
    return (loss_max + loss_mean + loss_min) / 3.0


def load_clean_fold_rows(base_dir=".", splits_dir="splits", fold=0):
    import pandas as pd
    from src.metadata import load_metadata_artifacts

    metadata = load_metadata_artifacts(base_dir, input_dir=splits_dir)
    master_df = metadata["master_df"]
    file_level_df = metadata["file_level_df"]

    split_cols = file_level_df[["file_id", "split_role", "cv_fold"]].copy()
    merged = master_df.merge(split_cols, on="file_id", how="left")
    merged = merged[
        (~merged["is_soundscape"])
        & (merged["phase_group"] == "clean")
        & (merged["split_role"] == "trainval")
    ].copy()

    train_rows = merged[merged["cv_fold"] != fold].copy()
    val_rows = merged[merged["cv_fold"] == fold].copy()
    return train_rows.to_dict("records"), val_rows.to_dict("records")


def evaluate_when(model, loader, device):
    model.eval()
    total_loss = 0.0
    all_max = []

    with torch.no_grad():
        progress = tqdm(loader, desc="when val", leave=False)
        for inputs, labels in progress:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(inputs)
            loss = mmm_loss(logits, labels)
            total_loss += loss.item() * inputs.size(0)
            all_max.append(torch.sigmoid(logits).max(dim=1).values.cpu().numpy())
            progress.set_postfix(loss=f"{loss.item():.4f}")

    return {
        "val_loss": total_loss / len(loader.dataset),
        "avg_max_prob": float(np.concatenate(all_max).mean()) if all_max else None,
    }


def train_when_detector(
    fold,
    base_dir=".",
    splits_dir="splits",
    output_dir="when_checkpoints",
    epochs=8,
    batch_size=16,
    lr=1e-3,
    num_workers=0,
    device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    train_rows, val_rows = load_clean_fold_rows(
        base_dir=base_dir,
        splits_dir=splits_dir,
        fold=fold,
    )
    if not train_rows or not val_rows:
        raise RuntimeError("No clean rows found for WHEN training.")

    train_loader = DataLoader(
        WhenClipDataset(train_rows, base_dir=base_dir, random_crop=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        WhenClipDataset(val_rows, base_dir=base_dir, random_crop=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    model = WhenNet(n_mels=N_MELS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(output_dir) / f"when_fold_{fold}_best.pt"

    best_val_loss = float("inf")
    best_state_dict = None
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        progress = tqdm(train_loader, desc=f"when train {epoch}/{epochs}", leave=False)

        for inputs, labels in progress:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = mmm_loss(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_loss /= len(train_loader.dataset)
        val_metrics = evaluate_when(model, val_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                **val_metrics,
            }
        )

        print(
            f"when fold={fold} epoch={epoch}/{epochs} "
            f"train_loss={train_loss:.5f} "
            f"val_loss={val_metrics['val_loss']:.5f} "
            f"avg_max_prob={val_metrics['avg_max_prob']:.5f}"
        )

        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            best_state_dict = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    "fold": fold,
                    "model_state_dict": model.state_dict(),
                    "history": history,
                    "best_val_loss": best_val_loss,
                    "n_mels": N_MELS,
                    "hop_length": HOP_LENGTH,
                    "n_fft": N_FFT,
                    "duration": DURATION,
                    "fmin": FMIN,
                    "fmax": FMAX,
                },
                checkpoint_path,
            )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return {
        "checkpoint_path": str(checkpoint_path),
        "best_val_loss": best_val_loss,
        "history": history,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
    }


def main():
    parser = argparse.ArgumentParser(description="Train WHEN detector on clean clips.")
    parser.add_argument("--base-dir", default=".")
    parser.add_argument("--splits-dir", default="splits")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--output-dir", default="when_checkpoints")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    result = train_when_detector(
        fold=args.fold,
        base_dir=args.base_dir,
        splits_dir=args.splits_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        device=args.device,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
