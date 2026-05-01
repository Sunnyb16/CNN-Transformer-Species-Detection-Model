import numpy as np
import librosa
import os
import soundfile as sf
from src.config import SR, N_FFT, HOP_LENGTH, N_MELS, FMIN, FMAX

#chunking data based on signal detection using fourier transform
SR = SR
DURATION = 5
CHUNK_SIZE = SR * DURATION


def moving_average(x, window):
    if window <= 1:
        return x.copy()

    pad_left = window // 2
    pad_right = window - 1 - pad_left
    x_pad = np.pad(x, (pad_left, pad_right), mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(x_pad, kernel, mode="valid")


def db(x, eps=1e-10):
    return 10 * np.log10(np.maximum(x, eps))


def merge_intervals(intervals, min_gap=0.25):
    if not intervals:
        return []

    intervals = sorted(intervals, key=lambda z: z[0])
    merged = [intervals[0]]

    for start, end in intervals[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= min_gap:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    return merged


def keep_long_enough(intervals, min_duration=0.15):
    return [(s, e) for s, e in intervals if (e - s) >= min_duration]


def pad_or_crop_to_fixed_length(y, sr, center_time, target_sec=5.0):
    target_len = int(round(target_sec * sr))
    center_sample = int(round(center_time * sr))

    start = center_sample - target_len // 2
    end = start + target_len

    pad_left = max(0, -start)
    pad_right = max(0, end - len(y))

    start_clipped = max(0, start)
    end_clipped = min(len(y), end)

    clip = y[start_clipped:end_clipped]

    if pad_left > 0 or pad_right > 0:
        clip = np.pad(clip, (pad_left, pad_right), mode="constant")

    if len(clip) > target_len:
        clip = clip[:target_len]
    elif len(clip) < target_len:
        clip = np.pad(clip, (0, target_len - len(clip)), mode="constant")

    return clip


def auto_find_frequency_band(
    S_power,
    freqs,
    fmin=300,
    fmax=None,
    smooth_bins=7,
    peak_rel_db=6.0,
    min_band_width_hz=500,
    max_band_width_hz=8000,
):
    if fmax is None:
        fmax = freqs[-1]

    valid = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(valid):
        raise ValueError("No frequency bins in requested range.")

    freqs_valid = freqs[valid]
    spectrum_mean = np.mean(S_power[valid, :], axis=1)
    spectrum_smooth = moving_average(spectrum_mean, smooth_bins)

    peak_idx = int(np.argmax(spectrum_smooth))
    peak_power = spectrum_smooth[peak_idx]

    threshold_power = peak_power / (10 ** (peak_rel_db / 10))

    left = peak_idx
    while left > 0 and spectrum_smooth[left] >= threshold_power:
        left -= 1

    right = peak_idx
    while right < len(spectrum_smooth) - 1 and spectrum_smooth[right] >= threshold_power:
        right += 1

    band_low = freqs_valid[left]
    band_high = freqs_valid[right]
    band_width = band_high - band_low

    if band_width < min_band_width_hz:
        extra = (min_band_width_hz - band_width) / 2
        band_low = max(fmin, band_low - extra)
        band_high = min(fmax, band_high + extra)

    if (band_high - band_low) > max_band_width_hz:
        center = (band_low + band_high) / 2
        half = max_band_width_hz / 2
        band_low = max(fmin, center - half)
        band_high = min(fmax, center + half)

    return float(band_low), float(band_high)


def detect_signal_intervals(
    y,
    sr,
    n_fft=2048,
    hop_length=256,
    fmin=300,
    fmax=None,
    band_peak_rel_db=6.0,
    energy_smooth_frames=9,
    background_smooth_frames=151,
    threshold_db=5.0,
    min_event_duration=0.15,
    merge_gap=0.25,
):
    if fmax is None:
        fmax = sr / 2


    # ensure valid STFT input

    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y)))


    # STFT


    # FIX 1: pad short signals
    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y)))


    # STFT

    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    S_power = np.abs(S) ** 2

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)


    # FIX 2: FORCE SHAPE MATCH

    if S_power.shape[0] != len(freqs):
        min_len = min(S_power.shape[0], len(freqs))
        S_power = S_power[:min_len, :]
        freqs = freqs[:min_len]


    # CRITICAL FIX: force shape match

    if S_power.shape[0] != len(freqs):
        min_len = min(S_power.shape[0], len(freqs))
        S_power = S_power[:min_len, :]
        freqs = freqs[:min_len]

    times = librosa.frames_to_time(np.arange(S_power.shape[1]), sr=sr, hop_length=hop_length)

    band_low, band_high = auto_find_frequency_band(
        S_power=S_power,
        freqs=freqs,
        fmin=fmin,
        fmax=fmax,
        peak_rel_db=band_peak_rel_db,
    )

    band_mask = (freqs >= band_low) & (freqs <= band_high)
    band_energy = np.sum(S_power[band_mask, :], axis=0)
    band_energy_smooth = moving_average(band_energy, energy_smooth_frames)

    local_background = moving_average(band_energy_smooth, background_smooth_frames)
    contrast_db = db(band_energy_smooth) - db(local_background)

    active = contrast_db > threshold_db

    intervals = []
    start_idx = None

    for i, flag in enumerate(active):
        if flag and start_idx is None:
            start_idx = i
        elif not flag and start_idx is not None:
            intervals.append((times[start_idx], times[i]))
            start_idx = None

    if start_idx is not None:
        intervals.append((times[start_idx], times[-1]))

    intervals = merge_intervals(intervals, min_gap=merge_gap)
    intervals = keep_long_enough(intervals, min_duration=min_event_duration)

    return {
        "intervals": intervals,
        "times": times,
        "contrast_db": contrast_db,
        "band_low_hz": band_low,
        "band_high_hz": band_high,
    }


def split_audio_fallback(y, sr, duration=5):
    chunk_size = int(sr * duration)
    chunks = []

    for i in range(0, len(y), chunk_size):
        chunk = y[i:i + chunk_size]

        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))

        chunks.append(chunk)

    return chunks


def split_audio_with_intervals(y, sr, duration=5):
    chunk_size = int(sr * duration)
    chunks = []
    intervals = []

    for start in range(0, len(y), chunk_size):
        end = start + chunk_size
        chunk = y[start:end]

        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))

        chunks.append(chunk)
        intervals.append((start / sr, end / sr))

    return {
        "chunks": chunks,
        "intervals": intervals,
        "used_fallback": True,
    }


def random_audio_chunks_from_array(
    y,
    sr,
    duration=5.0,
    seed=42,
):
    rng = np.random.default_rng(seed)
    chunk_size = int(sr * duration)

    if len(y) <= chunk_size:
        padded = np.pad(y, (0, max(0, chunk_size - len(y))))
        return {
            "chunks": [padded],
            "intervals": [(0.0, duration)],
            "used_fallback": False,
        }

    chunks = []
    intervals = []
    remaining_segments = [(0, len(y))]

    while True:
        valid_segments = [
            (start, end)
            for start, end in remaining_segments
            if (end - start) >= chunk_size
        ]
        if not valid_segments:
            break

        chosen_segment_idx = int(rng.integers(0, len(valid_segments)))
        seg_start, seg_end = valid_segments[chosen_segment_idx]
        max_start = seg_end - chunk_size
        start = int(rng.integers(seg_start, max_start + 1))
        end = start + chunk_size

        chunks.append(y[start:end])
        intervals.append((start / sr, end / sr))

        next_segments = []
        for cur_start, cur_end in remaining_segments:
            if end <= cur_start or start >= cur_end:
                next_segments.append((cur_start, cur_end))
                continue
            if cur_start < start:
                next_segments.append((cur_start, start))
            if end < cur_end:
                next_segments.append((end, cur_end))
        remaining_segments = next_segments

    return {
        "chunks": chunks,
        "intervals": intervals,
        "used_fallback": False,
    }

def get_signal_centered_chunks(
    audio_path,
    sr=SR,
    duration=DURATION,
    mono=True,
    n_fft=2048,
    hop_length=256,
    fmin=300,
    fmax=None,
    band_peak_rel_db=6.0,
    energy_smooth_frames=9,
    background_smooth_frames=151,
    threshold_db=5.0,
    min_event_duration=0.15,
    merge_gap=0.25,
    fallback_to_regular_split=True,
    only_strongest=True,
):
    y, sr_loaded = librosa.load(audio_path, sr=sr, mono=mono)

    return get_signal_centered_chunks_from_array(
        y=y,
        sr=sr_loaded,
        duration=duration,
        n_fft=n_fft,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax,
        band_peak_rel_db=band_peak_rel_db,
        energy_smooth_frames=energy_smooth_frames,
        background_smooth_frames=background_smooth_frames,
        threshold_db=threshold_db,
        min_event_duration=min_event_duration,
        merge_gap=merge_gap,
        fallback_to_regular_split=fallback_to_regular_split,
        only_strongest=only_strongest,
    )

def get_signal_centered_chunks_from_array(
    y,
    sr,
    duration,
    n_fft,
    hop_length,
    fmin,
    fmax,
    band_peak_rel_db,
    energy_smooth_frames,
    background_smooth_frames,
    threshold_db,
    min_event_duration,
    merge_gap,
    fallback_to_regular_split,
    only_strongest,
):

    eps = 1e-10

    # STFT requires len(y) >= n_fft
    # Without this → you get the "n_fft too large" + shape errors you saw
    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y)))


    # 1. Spectrogram

    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))**2

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    if len(freqs) != S.shape[0]:
        return {
            "chunks": [],
            "intervals": [],
            "used_fallback": True,
            "band_low_hz": fmin,
            "band_high_hz": fmax if fmax else sr // 2,
        }

    if fmax is None:
        fmax = sr // 2

    band_mask = (freqs >= fmin) & (freqs <= fmax)
    band_energy = S[band_mask].mean(axis=0)


    # 2. Convert to dB

    band_db = librosa.power_to_db(band_energy + eps)


    # 3. Smooth signal + background

    def moving_avg(x, k):
        return np.convolve(x, np.ones(k)/k, mode="same")

    energy_smooth = moving_avg(band_db, energy_smooth_frames)
    background = moving_avg(band_db, background_smooth_frames)

    rel_db = energy_smooth - background


    # 4. Detect active frames

    active = rel_db > threshold_db


    # 5. Convert to intervals

    intervals = []
    start = None

    for i, val in enumerate(active):
        if val and start is None:
            start = i
        elif not val and start is not None:
            end = i
            intervals.append((start, end))
            start = None

    if start is not None:
        intervals.append((start, len(active)))


    # 6. Convert frames → seconds

    intervals_sec = []
    for s, e in intervals:
        t_start = s * hop_length / sr
        t_end = e * hop_length / sr
        intervals_sec.append((t_start, t_end))


    # 7. Filter short events

    intervals_sec = [
        (s, e) for (s, e) in intervals_sec
        if (e - s) >= min_event_duration
    ]


    # 8. Merge close events

    merged = []
    for s, e in intervals_sec:
        if not merged:
            merged.append((s, e))
        else:
            prev_s, prev_e = merged[-1]
            if s - prev_e <= merge_gap:
                merged[-1] = (prev_s, e)
            else:
                merged.append((s, e))

    intervals_sec = merged


    # 9. Fallback if nothing found

    if len(intervals_sec) == 0:
        if fallback_to_regular_split:
            samples_per_chunk = int(sr * duration)
            chunks = []
            intervals_out = []

            for start in range(0, len(y), samples_per_chunk):
                end = start + samples_per_chunk
                chunk = y[start:end]

                if len(chunk) < samples_per_chunk:
                    chunk = np.pad(chunk, (0, samples_per_chunk - len(chunk)))

                chunks.append(chunk)
                intervals_out.append((start / sr, end / sr))

            return {
                "chunks": chunks,
                "intervals": intervals_out,
                "used_fallback": True,
                "band_low_hz": fmin,
                "band_high_hz": fmax,
            }
        else:
            return {
                "chunks": [],
                "intervals": [],
                "used_fallback": False,
                "band_low_hz": fmin,
                "band_high_hz": fmax,
            }


    # 10. Build centered chunks

    chunks = []
    chunk_intervals = []
    half = duration / 2

    if only_strongest:
        # pick strongest event
        energies = [(e - s, (s, e)) for (s, e) in intervals_sec]
        _, (s, e) = max(energies)
        center = (s + e) / 2

        start = max(0, center - half)
        end = start + duration

        start_sample = int(start * sr)
        end_sample = int(end * sr)

        chunk = y[start_sample:end_sample]

        if len(chunk) < int(sr * duration):
            chunk = np.pad(chunk, (0, int(sr * duration) - len(chunk)))

        chunks.append(chunk)
        chunk_intervals.append((start, end))

    else:
        for (s, e) in intervals_sec:
            center = (s + e) / 2

            start = max(0, center - half)
            end = start + duration

            start_sample = int(start * sr)
            end_sample = int(end * sr)

            chunk = y[start_sample:end_sample]

            if len(chunk) < int(sr * duration):
                chunk = np.pad(chunk, (0, int(sr * duration) - len(chunk)))

            chunks.append(chunk)
            chunk_intervals.append((start, end))

    return {
        "chunks": chunks,
        "intervals": chunk_intervals,
        "used_fallback": False,
        "band_low_hz": fmin,
        "band_high_hz": fmax,
    }

def audio_to_logmel(chunk, sr=SR):
    mel = librosa.feature.melspectrogram(
        y=chunk,
        sr=sr,
        n_fft=1024,
        hop_length=512,
        n_mels=128,
        fmin=20,
        fmax=8000
    )

    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)

    return log_mel.astype(np.float32)


def audio_file_to_logmels(
    audio_path,
    sr=SR,
    duration=DURATION,
    mono=True,
    only_strongest=False,
    fallback_to_regular_split=True,
    threshold_db=5.0,
    band_peak_rel_db=6.0,
    min_event_duration=0.15,
    merge_gap=0.25,
):
    result = get_signal_centered_chunks(
        audio_path=audio_path,
        sr=sr,
        duration=duration,
        mono=mono,
        only_strongest=only_strongest,
        fallback_to_regular_split=fallback_to_regular_split,
        threshold_db=threshold_db,
        band_peak_rel_db=band_peak_rel_db,
        min_event_duration=min_event_duration,
        merge_gap=merge_gap,
    )

    chunks = result["chunks"]
    logmels = [audio_to_logmel(chunk, sr=sr) for chunk in chunks]

    return {
        "logmels": logmels,
        "chunks": chunks,
        "intervals": result["intervals"],
        "band_low_hz": result["band_low_hz"],
        "band_high_hz": result["band_high_hz"],
        "used_fallback": result["used_fallback"],
    }


def save_chunks(chunks, sr, output_dir, base_name="audio"):
    os.makedirs(output_dir, exist_ok=True)

    for i, chunk in enumerate(chunks):
        out_path = os.path.join(output_dir, f"{base_name}_chunk_{i:03d}.wav")
        sf.write(out_path, chunk, sr)

def get_chunks_for_inference(
    audio_path,
    sr,
    duration,
    **chunk_config
):
    result = get_signal_centered_chunks(
        audio_path=audio_path,
        sr=sr,
        duration=duration,
        **chunk_config
    )

    return result["chunks"]
