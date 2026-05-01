import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import torch

from src.audio_processing import audio_file_to_logmels, audio_to_logmel, split_audio_with_intervals
from src.config import CHUNK_CONFIG, DURATION, SR
from src.model import BirdResNet
from src.paths import resolve_path


def load_checkpoint(checkpoint_path, device, base_dir="."):
    checkpoint = torch.load(resolve_path(checkpoint_path, base_dir), map_location=device)

    if "class_list" not in checkpoint or "model_state_dict" not in checkpoint:
        raise ValueError(
            "Checkpoint must contain 'class_list' and 'model_state_dict'. "
            "Train with the new src.train pipeline first."
        )

    dropout = checkpoint.get("dropout", 0.2)
    model = BirdResNet(num_classes=len(checkpoint["class_list"]), dropout=dropout).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    species_class_list = checkpoint.get("species_class_list", checkpoint["class_list"])
    return model, checkpoint["class_list"], species_class_list


def _should_use_soundscape_windows(audio_path, mode, duration_sec):
    if mode == "soundscape":
        return True
    if mode == "clip":
        return False

    audio_name = Path(audio_path).name.lower()
    if "soundscape" in audio_name or audio_name.startswith("bc2026_test_"):
        return True

    try:
        audio_duration = librosa.get_duration(path=audio_path)
    except Exception:
        audio_duration = None

    return audio_duration is not None and audio_duration > (2.0 * duration_sec)


def _load_logmels_for_soundscape(audio_path, duration_sec, hop_sec):
    y, _ = librosa.load(audio_path, sr=SR, mono=True)

    if hop_sec <= 0:
        raise ValueError("hop_sec must be > 0.")

    chunk_size = int(round(SR * duration_sec))
    hop_size = int(round(SR * hop_sec))
    if hop_size <= 0:
        raise ValueError("hop_sec produced a non-positive hop size.")

    chunks = []
    intervals = []

    if len(y) <= chunk_size:
        padded = np.pad(y, (0, max(0, chunk_size - len(y))))
        chunks = [padded]
        intervals = [(0.0, duration_sec)]
    elif hop_size == chunk_size:
        split_result = split_audio_with_intervals(y=y, sr=SR, duration=duration_sec)
        chunks = split_result["chunks"]
        intervals = split_result["intervals"]
    else:
        for start in range(0, len(y), hop_size):
            end = start + chunk_size
            chunk = y[start:end]
            if len(chunk) == 0:
                continue
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
            chunks.append(chunk)
            intervals.append((start / SR, end / SR))
            if end >= len(y):
                break

    logmels = [audio_to_logmel(chunk, sr=SR) for chunk in chunks]
    return logmels, intervals


def _aggregate_chunk_probs(probs, aggregation):
    if aggregation == "mean":
        return probs.mean(axis=0)
    if aggregation == "meanmax":
        return 0.5 * (probs.max(axis=0) + probs.mean(axis=0))
    return probs.max(axis=0)


@torch.no_grad()
def predict_file(
    audio_path,
    checkpoint_path,
    top_k=10,
    device=None,
    base_dir=".",
    inference_mode="auto",
    window_duration=DURATION,
    window_hop=DURATION,
    aggregation="max",
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    audio_path = resolve_path(audio_path, base_dir)
    model, full_class_list, species_class_list = load_checkpoint(
        checkpoint_path,
        device,
        base_dir=base_dir,
    )

    use_soundscape_windows = _should_use_soundscape_windows(
        audio_path=audio_path,
        mode=inference_mode,
        duration_sec=window_duration,
    )

    intervals = None
    if use_soundscape_windows:
        logmels, intervals = _load_logmels_for_soundscape(
            audio_path=audio_path,
            duration_sec=window_duration,
            hop_sec=window_hop,
        )
    else:
        chunk_result = audio_file_to_logmels(
            audio_path=audio_path,
            sr=SR,
            duration=window_duration,
            only_strongest=False,
            fallback_to_regular_split=True,
            **CHUNK_CONFIG,
        )
        logmels = chunk_result["logmels"]
        intervals = chunk_result.get("intervals")

    if not logmels:
        raise RuntimeError(f"No chunks were produced for {audio_path}")

    batch = torch.from_numpy(np.stack(logmels)).unsqueeze(1).to(device)
    logits = model(batch)
    probs = torch.sigmoid(logits).cpu().numpy()
    aggregated = _aggregate_chunk_probs(probs, aggregation=aggregation)
    aggregated_species = aggregated[:len(species_class_list)]

    top_indices = np.argsort(aggregated_species)[::-1][:top_k]
    predictions = [
        {"label": species_class_list[idx], "score": float(aggregated_species[idx])}
        for idx in top_indices
    ]

    return {
        "audio_path": audio_path,
        "num_chunks": len(logmels),
        "inference_mode": "soundscape" if use_soundscape_windows else "clip",
        "window_duration": window_duration,
        "window_hop": window_hop,
        "aggregation": aggregation,
        "intervals": intervals,
        "predictions": predictions,
    }


def main():
    parser = argparse.ArgumentParser(description="Run BirdCLEF inference for one file.")
    parser.add_argument("audio_path")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", default=None)
    parser.add_argument("--base-dir", default=".")
    parser.add_argument("--inference-mode", choices=["auto", "clip", "soundscape"], default="auto")
    parser.add_argument("--window-duration", type=float, default=DURATION)
    parser.add_argument("--window-hop", type=float, default=DURATION)
    parser.add_argument("--aggregation", choices=["max", "mean", "meanmax"], default="max")
    args = parser.parse_args()

    result = predict_file(
        audio_path=args.audio_path,
        checkpoint_path=args.checkpoint,
        top_k=args.top_k,
        device=args.device,
        base_dir=args.base_dir,
        inference_mode=args.inference_mode,
        window_duration=args.window_duration,
        window_hop=args.window_hop,
        aggregation=args.aggregation,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
