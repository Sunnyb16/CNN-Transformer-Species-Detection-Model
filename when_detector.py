from pathlib import Path

import librosa
import numpy as np
import torch

from src.audio_processing import pad_or_crop_to_fixed_length
from src.config import DURATION, FMAX, FMIN, HOP_LENGTH, N_FFT, N_MELS, SR
from src.paths import resolve_path
from src.when_model import WhenNet


_WHEN_MODEL = None
_WHEN_MODEL_PATH = None
_WHEN_DEVICE = None
_WHEN_CONFIG = None


def load_when_detector(checkpoint_path, device=None, base_dir="."):
    global _WHEN_MODEL, _WHEN_MODEL_PATH, _WHEN_DEVICE, _WHEN_CONFIG

    resolved_path = str(resolve_path(checkpoint_path, base_dir))
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if (
        _WHEN_MODEL is not None
        and _WHEN_MODEL_PATH == resolved_path
        and _WHEN_DEVICE == device
    ):
        return _WHEN_MODEL, _WHEN_CONFIG

    checkpoint = torch.load(resolved_path, map_location=device)
    model = WhenNet(n_mels=checkpoint.get("n_mels", N_MELS)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _WHEN_MODEL = model
    _WHEN_MODEL_PATH = resolved_path
    _WHEN_DEVICE = device
    _WHEN_CONFIG = {
        "n_mels": checkpoint.get("n_mels", N_MELS),
        "hop_length": checkpoint.get("hop_length", HOP_LENGTH),
        "n_fft": checkpoint.get("n_fft", N_FFT),
        "duration": checkpoint.get("duration", DURATION),
        "fmin": checkpoint.get("fmin", FMIN),
        "fmax": checkpoint.get("fmax", FMAX),
    }
    return _WHEN_MODEL, _WHEN_CONFIG


def audio_to_when_logmel(chunk, config):
    mel = librosa.feature.melspectrogram(
        y=chunk,
        sr=SR,
        n_fft=config["n_fft"],
        hop_length=config["hop_length"],
        n_mels=config["n_mels"],
        fmin=config["fmin"],
        fmax=config["fmax"],
    )
    log_mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
    return log_mel


def predict_event_probs_over_audio(
    y,
    checkpoint_path,
    base_dir=".",
    device=None,
    window_sec=5.0,
    hop_sec=5.0,
):
    model, config = load_when_detector(
        checkpoint_path=checkpoint_path,
        device=device,
        base_dir=base_dir,
    )

    window_len = int(round(SR * window_sec))
    hop_len = int(round(SR * hop_sec))
    if hop_len <= 0:
        raise ValueError("hop_sec must be > 0.")

    windows = []
    offsets = []

    if len(y) <= window_len:
        windows = [np.pad(y, (0, max(0, window_len - len(y))))]
        offsets = [0]
    else:
        for start in range(0, len(y), hop_len):
            end = start + window_len
            chunk = y[start:end]
            if len(chunk) == 0:
                continue
            if len(chunk) < window_len:
                chunk = np.pad(chunk, (0, window_len - len(chunk)))
            windows.append(chunk)
            offsets.append(start)
            if end >= len(y):
                break

    specs = [audio_to_when_logmel(chunk, config) for chunk in windows]
    batch = torch.from_numpy(np.stack(specs)).unsqueeze(1).to(_WHEN_DEVICE)

    with torch.no_grad():
        probs = torch.sigmoid(model(batch)).cpu().numpy()

    frame_hop_sec = config["hop_length"] / SR
    global_probs = []
    global_times = []

    for offset, prob_seq in zip(offsets, probs):
        for frame_idx, prob in enumerate(prob_seq):
            global_probs.append(float(prob))
            global_times.append((offset / SR) + (frame_idx * frame_hop_sec))

    return np.asarray(global_probs, dtype=np.float32), np.asarray(global_times, dtype=np.float32)


def group_event_probs(
    probs,
    times,
    threshold=0.6,
    min_event_duration=0.15,
    merge_gap=0.25,
):
    if len(probs) == 0:
        return []

    active = probs >= threshold
    intervals = []
    start_idx = None

    for idx, flag in enumerate(active):
        if flag and start_idx is None:
            start_idx = idx
        elif not flag and start_idx is not None:
            end_idx = idx - 1
            intervals.append((times[start_idx], times[end_idx]))
            start_idx = None

    if start_idx is not None:
        intervals.append((times[start_idx], times[-1]))

    merged = []
    for start, end in intervals:
        if not merged:
            merged.append((start, end))
            continue
        prev_start, prev_end = merged[-1]
        if start - prev_end <= merge_gap:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))

    return [
        (start, end)
        for start, end in merged
        if (end - start) >= min_event_duration
    ]


def get_when_centered_chunks_from_array(
    y,
    checkpoint_path,
    base_dir=".",
    device=None,
    target_chunk_sec=DURATION,
    when_threshold=0.6,
    min_event_duration=0.15,
    merge_gap=0.25,
    fallback_to_regular_split=True,
):
    probs, times = predict_event_probs_over_audio(
        y=y,
        checkpoint_path=checkpoint_path,
        base_dir=base_dir,
        device=device,
        window_sec=target_chunk_sec,
        hop_sec=target_chunk_sec,
    )
    events = group_event_probs(
        probs=probs,
        times=times,
        threshold=when_threshold,
        min_event_duration=min_event_duration,
        merge_gap=merge_gap,
    )

    if not events:
        if fallback_to_regular_split:
            chunk = pad_or_crop_to_fixed_length(
                y=y,
                sr=SR,
                center_time=max(0.0, len(y) / (2 * SR)),
                target_sec=target_chunk_sec,
            )
            return {
                "chunks": [chunk],
                "intervals": [(0.0, target_chunk_sec)],
                "used_fallback": True,
                "avg_event_duration_sec": None,
                "event_intervals": [],
            }
        return {
            "chunks": [],
            "intervals": [],
            "used_fallback": False,
            "avg_event_duration_sec": None,
            "event_intervals": [],
        }

    chunks = []
    chunk_intervals = []
    event_durations = []

    for start, end in events:
        center = (start + end) / 2.0
        chunk = pad_or_crop_to_fixed_length(
            y=y,
            sr=SR,
            center_time=center,
            target_sec=target_chunk_sec,
        )
        chunk_start = max(0.0, center - (target_chunk_sec / 2.0))
        chunk_end = chunk_start + target_chunk_sec

        chunks.append(chunk)
        chunk_intervals.append((chunk_start, chunk_end))
        event_durations.append(end - start)

    return {
        "chunks": chunks,
        "intervals": chunk_intervals,
        "used_fallback": False,
        "avg_event_duration_sec": float(np.mean(event_durations)) if event_durations else None,
        "event_intervals": events,
    }
