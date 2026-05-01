import torch
from torch.utils.data import Dataset
import numpy as np
import librosa
import random
import math

from src.config import SR, DURATION, N_FFT, HOP_LENGTH, N_MELS, FMIN, FMAX
from src.paths import resolve_path


class BirdChunkDataset(Dataset):
    def __init__(
        self,
        samples,
        augment=False,
        aug_params=None,
        sr=SR,
        base_dir=".",
        label_to_idx=None,
        secondary_label_total_weight=None,
    ):
        self.samples = samples
        self.augment = augment
        self.aug_params = aug_params or {}
        self.sr = sr
        self.base_dir = base_dir
        self.label_to_idx = label_to_idx or {}
        self.secondary_label_total_weight = secondary_label_total_weight

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # fast path (no augmentation)
        if not self.augment:
            spec = np.load(resolve_path(sample["spec_path"], self.base_dir), mmap_mode="r")

            spec = spec.astype(np.float32)
            spec = (spec - spec.mean()) / (spec.std() + 1e-6)

            spec = torch.from_numpy(spec).unsqueeze(0)
            target = self._build_target_tensor(sample)

            return spec, target

        # augmentation path
        audio_path = sample.get("audio_path") or sample.get("source_path")
        if audio_path is None:
            raise KeyError("Augmentation requires 'audio_path' or 'source_path' in sample.")
        audio_path = resolve_path(audio_path, self.base_dir)

        load_kwargs = {"sr": self.sr, "mono": True}
        source_interval = sample.get("source_interval")
        if (
            isinstance(source_interval, (list, tuple))
            and len(source_interval) == 2
            and all(v is not None and not math.isnan(v) for v in source_interval)
        ):
            start_time, end_time = source_interval
            load_kwargs["offset"] = float(start_time)
            load_kwargs["duration"] = float(end_time - start_time)

        y, _ = librosa.load(audio_path, **load_kwargs)

        y = augment_audio(y, sr=self.sr, duration=DURATION, **self.aug_params)

        mel = librosa.feature.melspectrogram(
            y=y,
            sr=self.sr,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            fmin=FMIN,
            fmax=FMAX,
        )

        log_mel = librosa.power_to_db(mel)
        log_mel = apply_spec_augment(log_mel, **self.aug_params)
        log_mel = log_mel.astype(np.float32)
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)

        spec = torch.from_numpy(log_mel).unsqueeze(0)
        target = self._build_target_tensor(sample)

        return spec, target

    def _build_target_tensor(self, sample):
        base_target = np.asarray(sample["target"], dtype=np.float32)

        if (
            self.secondary_label_total_weight is None
            or self.secondary_label_total_weight < 0
            or not self.label_to_idx
        ):
            return torch.tensor(base_target, dtype=torch.float32)

        labels = [
            label for label in sample.get("labels", [])
            if isinstance(label, str) and label in self.label_to_idx
        ]
        if not labels:
            return torch.tensor(base_target, dtype=torch.float32)

        weighted_target = base_target.copy()
        for label in labels:
            weighted_target[self.label_to_idx[label]] = 0.0

        primary_label = labels[0]
        weighted_target[self.label_to_idx[primary_label]] = 1.0

        secondary_labels = labels[1:]
        if secondary_labels:
            per_secondary_weight = (
                float(self.secondary_label_total_weight) / len(secondary_labels)
            )
            for label in secondary_labels:
                weighted_target[self.label_to_idx[label]] = per_secondary_weight

        return torch.tensor(weighted_target, dtype=torch.float32)


def augment_audio(
    y,
    sr,
    duration=5,
    pitch_prob=0.4,
    pitch_range=(-1.5, 1.5),
    stretch_prob=0.4,
    stretch_range=(0.9, 1.1),
    shift_prob=0.5,
    shift_max_sec=1.0,
    noise_prob=0.5,
    noise_std=0.005,
):
    target_len = int(sr * duration)

    # Normalize short/raw segments before spectral augment ops like
    # pitch_shift/time_stretch, which internally run STFT and expect
    # a reasonably sized waveform.
    y = pad_crop_audio(y, target_len)

    if random.random() < pitch_prob:
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=random.uniform(*pitch_range))

    if random.random() < stretch_prob:
        y = librosa.effects.time_stretch(y, rate=random.uniform(*stretch_range))

    if random.random() < shift_prob:
        max_shift = int(shift_max_sec * sr)
        shift = int(random.uniform(-max_shift, max_shift))
        y = np.roll(y, shift)
        if shift > 0:
            y[:shift] = 0
        else:
            y[shift:] = 0

    if random.random() < noise_prob:
        y = y + np.random.randn(len(y)) * noise_std

    return pad_crop_audio(y, target_len)


def apply_spec_augment(
    spec,
    time_mask_prob=0.0,
    time_mask_param=12,
    num_time_masks=1,
    freq_mask_prob=0.0,
    freq_mask_param=12,
    num_freq_masks=1,
    **_,
):
    augmented = np.array(spec, copy=True)
    n_mels, n_frames = augmented.shape

    if time_mask_prob > 0 and n_frames > 1 and random.random() < time_mask_prob:
        max_width = max(1, min(int(time_mask_param), n_frames))
        for _ in range(max(0, int(num_time_masks))):
            width = random.randint(1, max_width)
            start = random.randint(0, max(0, n_frames - width))
            augmented[:, start:start + width] = 0.0

    if freq_mask_prob > 0 and n_mels > 1 and random.random() < freq_mask_prob:
        max_width = max(1, min(int(freq_mask_param), n_mels))
        for _ in range(max(0, int(num_freq_masks))):
            width = random.randint(1, max_width)
            start = random.randint(0, max(0, n_mels - width))
            augmented[start:start + width, :] = 0.0

    return augmented


def pad_crop_audio(y, target_len):
    if len(y) < target_len:
        return np.pad(y, (0, target_len - len(y)))
    start = np.random.randint(0, len(y) - target_len + 1)
    return y[start:start + target_len]
