import os
import pickle
import shutil
import time
from functools import partial
from pathlib import Path

import librosa
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from src.audio_processing import (
    get_signal_centered_chunks_from_array,
    pad_or_crop_to_fixed_length,
    random_audio_chunks_from_array,
    split_audio_with_intervals,
)
from src.config import CHUNK_CONFIG, DURATION, FMAX, FMIN, HOP_LENGTH, N_FFT, N_MELS, SR
from src.labels import labels_to_taxonomy_labels
from src.paths import relativize_to_base, resolve_path
from src.templates import filter_secondary_labels_for_chunk
from src.when_detector import get_when_centered_chunks_from_array


TEMPLATES = None


TIMING_KEYS = (
    "resolve_path_sec",
    "chunk_extract_sec",
    "mel_sec",
    "template_filter_sec",
    "bundle_write_sec",
    "total_row_sec",
)


def _empty_timing_stats():
    return {
        "rows_seen": 0,
        "rows_with_samples": 0,
        "chunks_generated": 0,
        **{key: 0.0 for key in TIMING_KEYS},
    }


def _merge_timing_stats(accum, update):
    for key, value in update.items():
        if key in ("rows_seen", "rows_with_samples", "chunks_generated"):
            accum[key] += int(value)
        else:
            accum[key] += float(value)
    return accum


def summarize_timing_stats(timing_stats):
    if timing_stats is None:
        return None

    rows = []
    for split_name, stats in timing_stats.items():
        row = {"split": split_name}
        rows_seen = max(1, int(stats.get("rows_seen", 0)))
        chunks_generated = max(1, int(stats.get("chunks_generated", 0)))
        row.update(stats)
        row["avg_total_row_sec"] = float(stats.get("total_row_sec", 0.0)) / rows_seen
        row["avg_chunk_extract_sec"] = float(stats.get("chunk_extract_sec", 0.0)) / rows_seen
        row["avg_mel_sec_per_chunk"] = float(stats.get("mel_sec", 0.0)) / chunks_generated
        row["avg_template_filter_sec_per_chunk"] = (
            float(stats.get("template_filter_sec", 0.0)) / chunks_generated
        )
        row["avg_bundle_write_sec"] = float(stats.get("bundle_write_sec", 0.0)) / rows_seen
        rows.append(row)

    return rows


def _valid_time_bounds(row):
    start_time = row.get("start_time")
    end_time = row.get("end_time")

    if start_time is None or end_time is None:
        return False
    if np.isnan(start_time) or np.isnan(end_time):
        return False
    return end_time > start_time


def _resolve_effective_chunking_mode(row, chunking_mode, when_checkpoint_path=None):
    if chunking_mode != "hybrid":
        return chunking_mode

    phase_group = row.get("phase_group")
    if bool(row.get("is_soundscape", False)) or phase_group == "semi":
        return "sequential"
    if phase_group == "clean" and when_checkpoint_path:
        return "when"

    return "signal"


def _load_audio_cached(path, sr, audio_cache=None):
    if audio_cache is not None and path in audio_cache:
        return audio_cache[path]

    y, sr_loaded = librosa.load(path, sr=sr, mono=True)
    if audio_cache is not None:
        audio_cache[path] = (y, sr_loaded)
    return y, sr_loaded


def extract_row_chunks(
    row,
    base_dir,
    sr,
    duration,
    threshold_db,
    band_peak_rel_db,
    min_event_duration,
    merge_gap,
    only_strongest,
    fallback_to_regular_split,
    chunking_mode="signal",
    random_seed=42,
    when_checkpoint_path=None,
    audio_cache=None,
):
    path = resolve_path(row["full_path"], base_dir)
    y_full, sr_loaded = _load_audio_cached(path, sr=sr, audio_cache=audio_cache)
    effective_chunking_mode = _resolve_effective_chunking_mode(
        row=row,
        chunking_mode=chunking_mode,
        when_checkpoint_path=when_checkpoint_path,
    )

    if _valid_time_bounds(row):
        start_sample = int(round(float(row["start_time"]) * sr_loaded))
        end_sample = int(round(float(row["end_time"]) * sr_loaded))
        y = y_full[start_sample:end_sample]

        if bool(row.get("is_soundscape", False)):
            fixed_chunk = pad_or_crop_to_fixed_length(
                y=y,
                sr=sr_loaded,
                center_time=max(0.0, len(y) / (2 * sr_loaded)),
                target_sec=duration,
            )
            chunk_end = float(row["start_time"]) + duration
            return {
                "chunks": [fixed_chunk],
                "intervals": [(float(row["start_time"]), chunk_end)],
                "used_fallback": False,
            }

        if effective_chunking_mode == "sequential":
            result = split_audio_with_intervals(y=y, sr=sr_loaded, duration=duration)
        elif effective_chunking_mode == "when":
            result = get_when_centered_chunks_from_array(
                y=y,
                checkpoint_path=when_checkpoint_path,
                base_dir=base_dir,
                target_chunk_sec=duration,
                min_event_duration=min_event_duration,
                merge_gap=merge_gap,
                fallback_to_regular_split=fallback_to_regular_split,
            )
        elif effective_chunking_mode == "random":
            result = random_audio_chunks_from_array(
                y=y,
                sr=sr_loaded,
                duration=duration,
                seed=random_seed,
            )
        else:
            result = get_signal_centered_chunks_from_array(
                y=y,
                sr=sr_loaded,
                duration=duration,
                n_fft=2048,
                hop_length=256,
                fmin=300,
                fmax=None,
                band_peak_rel_db=band_peak_rel_db,
                energy_smooth_frames=9,
                background_smooth_frames=151,
                threshold_db=threshold_db,
                min_event_duration=min_event_duration,
                merge_gap=merge_gap,
                fallback_to_regular_split=fallback_to_regular_split,
                only_strongest=only_strongest,
            )

        # Re-anchor chunk intervals to absolute file time.
        anchored_intervals = []
        for start, end in result["intervals"]:
            anchored_intervals.append(
                (float(row["start_time"]) + start, float(row["start_time"]) + end)
            )

        result["intervals"] = anchored_intervals
        return result

    y = y_full
    if effective_chunking_mode == "sequential":
        return split_audio_with_intervals(y=y, sr=sr_loaded, duration=duration)
    if effective_chunking_mode == "when":
        return get_when_centered_chunks_from_array(
            y=y,
            checkpoint_path=when_checkpoint_path,
            base_dir=base_dir,
            target_chunk_sec=duration,
            min_event_duration=min_event_duration,
            merge_gap=merge_gap,
            fallback_to_regular_split=fallback_to_regular_split,
        )
    if effective_chunking_mode == "random":
        return random_audio_chunks_from_array(
            y=y,
            sr=sr_loaded,
            duration=duration,
            seed=random_seed,
        )
    return get_signal_centered_chunks_from_array(
        y=y,
        sr=sr_loaded,
        duration=duration,
        n_fft=2048,
        hop_length=256,
        fmin=300,
        fmax=None,
        band_peak_rel_db=band_peak_rel_db,
        energy_smooth_frames=9,
        background_smooth_frames=151,
        threshold_db=threshold_db,
        min_event_duration=min_event_duration,
        merge_gap=merge_gap,
        fallback_to_regular_split=fallback_to_regular_split,
        only_strongest=only_strongest,
    )


def process_single_row(
    row,
    base_dir,
    output_dir,
    encode_labels,
    sr,
    duration,
    threshold_db,
    band_peak_rel_db,
    min_event_duration,
    merge_gap,
    only_strongest,
    fallback_to_regular_split,
    chunking_mode,
    random_seed,
    taxonomy_lookup=None,
    when_checkpoint_path=None,
    collect_timing=False,
    audio_cache=None,
):
    global TEMPLATES
    row_start = time.perf_counter()
    timing = _empty_timing_stats()
    timing["rows_seen"] = 1

    t0 = time.perf_counter()
    path = resolve_path(row["full_path"], base_dir)
    timing["resolve_path_sec"] += time.perf_counter() - t0
    if not isinstance(path, str) or not os.path.exists(path):
        timing["total_row_sec"] += time.perf_counter() - row_start
        return ([], timing) if collect_timing else []

    try:
        t0 = time.perf_counter()
        result = extract_row_chunks(
            row=row,
            base_dir=base_dir,
            sr=sr,
            duration=duration,
            threshold_db=threshold_db,
            band_peak_rel_db=band_peak_rel_db,
            min_event_duration=min_event_duration,
            merge_gap=merge_gap,
            only_strongest=only_strongest,
            fallback_to_regular_split=fallback_to_regular_split,
            chunking_mode=chunking_mode,
            random_seed=random_seed,
            when_checkpoint_path=when_checkpoint_path,
            audio_cache=audio_cache,
        )
        timing["chunk_extract_sec"] += time.perf_counter() - t0
    except Exception:
        timing["total_row_sec"] += time.perf_counter() - row_start
        return ([], timing) if collect_timing else []

    chunks = result["chunks"]
    intervals = result.get("intervals", [])
    used_fallback = bool(result.get("used_fallback", False))
    effective_chunking_mode = _resolve_effective_chunking_mode(
        row=row,
        chunking_mode=chunking_mode,
        when_checkpoint_path=when_checkpoint_path,
    )

    if effective_chunking_mode == "signal":
        chunk_origin = "signal_fallback" if used_fallback else "signal"
    else:
        chunk_origin = effective_chunking_mode
    if not chunks:
        timing["total_row_sec"] += time.perf_counter() - row_start
        return ([], timing) if collect_timing else []

    primary_label = row["primary_label"]
    secondary_labels = row.get("secondary_labels", [])
    base_name = os.path.splitext(os.path.basename(path))[0]

    file_dir = os.path.join(output_dir, base_name)
    os.makedirs(file_dir, exist_ok=True)

    samples = []
    for i, chunk in enumerate(chunks):
        try:
            t0 = time.perf_counter()
            mel = librosa.feature.melspectrogram(
                y=chunk,
                sr=sr,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
                n_mels=N_MELS,
                fmin=FMIN,
                fmax=FMAX,
            )
            mel_db = librosa.power_to_db(mel).astype(np.float32)
            timing["mel_sec"] += time.perf_counter() - t0

            filtered_secondary = list(secondary_labels)
            if TEMPLATES and secondary_labels and not _valid_time_bounds(row):
                candidate_overlap_sec = None
                t0 = time.perf_counter()
                filtered_secondary, _ = filter_secondary_labels_for_chunk(
                    chunk=chunk,
                    secondary_labels=secondary_labels,
                    templates=TEMPLATES,
                    candidate_overlap_sec=candidate_overlap_sec,
                    sr=sr,
                )
                timing["template_filter_sec"] += time.perf_counter() - t0

            labels = [primary_label] + filtered_secondary
            labels = [label for label in labels if isinstance(label, str) and label]
            taxonomy_labels = labels_to_taxonomy_labels(labels, taxonomy_lookup or {})
            target = encode_labels(labels + taxonomy_labels)

            spec_path = os.path.join(
                file_dir,
                f"{base_name}_{i}_{np.random.randint(1e9)}.npy",
            )
            t0 = time.perf_counter()
            np.save(spec_path, mel_db)
            timing["bundle_write_sec"] += time.perf_counter() - t0

            samples.append(
                {
                    "spec_path": relativize_to_base(spec_path, base_dir),
                    "target": target,
                    "source_path": relativize_to_base(path, base_dir),
                    "source_interval": intervals[i] if i < len(intervals) else None,
                    "used_fallback": used_fallback,
                    "chunk_origin": chunk_origin,
                    "labels": labels,
                    "taxonomy_labels": taxonomy_labels,
                    "avg_event_duration_sec": result.get("avg_event_duration_sec"),
                    "phase_group": row.get("phase_group"),
                    "file_id": row.get("file_id"),
                }
            )
        except Exception:
            continue

    if not samples:
        timing["total_row_sec"] += time.perf_counter() - row_start
        return ([], timing) if collect_timing else []

    timing["rows_with_samples"] = 1
    timing["chunks_generated"] = len(samples)
    timing["total_row_sec"] += time.perf_counter() - row_start
    return (samples, timing) if collect_timing else samples


def process_chunk(chunk_df, process_fn):
    outputs = []
    audio_cache = {}
    for _, row in chunk_df.iterrows():
        result = process_fn(row, audio_cache=audio_cache)
        if result:
            outputs.append(result)
    return outputs


def precompute_chunk_cache(
    df,
    base_dir,
    output_dir,
    encode_labels,
    sr,
    duration,
    threshold_db,
    band_peak_rel_db,
    min_event_duration,
    merge_gap,
    only_strongest=False,
    fallback_to_regular_split=True,
    chunking_mode="signal",
    random_seed=42,
    taxonomy_lookup=None,
    when_checkpoint_path=None,
    n_jobs=3,
    collect_timing=False,
):
    output_dir = str(Path(resolve_path(output_dir, base_dir)))
    os.makedirs(output_dir, exist_ok=True)
    df = df.sort_values(["full_path", "file_id"], kind="stable").copy()

    process_fn = partial(
        process_single_row,
        base_dir=base_dir,
        output_dir=output_dir,
        encode_labels=encode_labels,
        sr=sr,
        duration=duration,
        threshold_db=threshold_db,
        band_peak_rel_db=band_peak_rel_db,
        min_event_duration=min_event_duration,
        merge_gap=merge_gap,
        only_strongest=only_strongest,
        fallback_to_regular_split=fallback_to_regular_split,
        chunking_mode=chunking_mode,
        random_seed=random_seed,
        taxonomy_lookup=taxonomy_lookup,
        when_checkpoint_path=when_checkpoint_path,
        collect_timing=collect_timing,
    )

    num_splits = max(1, n_jobs * 6)
    index_splits = np.array_split(np.arange(len(df)), num_splits)
    work_chunks = [df.iloc[idx].copy() for idx in index_splits if len(idx) > 0]
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_chunk)(chunk_df, process_fn)
        for chunk_df in tqdm(work_chunks, desc="Processing chunks")
    )

    timing_stats = _empty_timing_stats() if collect_timing else None
    samples = []
    for chunk_result in results:
        for item in chunk_result:
            if collect_timing:
                sample_list, row_timing = item
                _merge_timing_stats(timing_stats, row_timing)
            else:
                sample_list = item
            samples.extend(sample_list)

    if not samples:
        raise RuntimeError("No samples generated during precompute.")

    if collect_timing:
        return samples, timing_stats
    return samples


def get_rows_for_files(master_df, file_df):
    file_ids = set(file_df["file_id"])
    return master_df[master_df["file_id"].isin(file_ids)].copy()


def precompute_fold(
    fold,
    file_level_df,
    master_df,
    templates,
    encode_labels,
    base_dir=".",
    base_output_dir="data_processed",
    duration=DURATION,
    n_jobs=3,
    force=False,
    chunking_mode="signal",
    random_seed=42,
    taxonomy_lookup=None,
    when_checkpoint_path=None,
    collect_timing=False,
):
    global TEMPLATES
    TEMPLATES = templates

    base_output_dir = str(Path(resolve_path(base_output_dir, base_dir)))
    fold_dir = os.path.join(base_output_dir, f"fold_{fold}")
    train_dir = os.path.join(fold_dir, "train")
    val_dir = os.path.join(fold_dir, "val")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    train_samples_path = os.path.join(fold_dir, "train_samples.pkl")
    val_samples_path = os.path.join(fold_dir, "val_samples.pkl")

    if force and os.path.isdir(fold_dir):
        shutil.rmtree(fold_dir)
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

    if os.path.exists(train_samples_path) and os.path.exists(val_samples_path):
        with open(train_samples_path, "rb") as handle:
            train_samples = pickle.load(handle)
        with open(val_samples_path, "rb") as handle:
            val_samples = pickle.load(handle)
        if collect_timing:
            return train_samples, val_samples, None
        return train_samples, val_samples

    train_files = file_level_df[
        (file_level_df["split_role"] == "trainval") & (file_level_df["cv_fold"] != fold)
    ]
    val_files = file_level_df[
        (file_level_df["split_role"] == "trainval") & (file_level_df["cv_fold"] == fold)
    ]

    train_rows = get_rows_for_files(master_df, train_files)
    val_rows = get_rows_for_files(master_df, val_files)

    train_result = precompute_chunk_cache(
        df=train_rows,
        base_dir=base_dir,
        output_dir=train_dir,
        encode_labels=encode_labels,
        sr=SR,
        duration=duration,
        chunking_mode=chunking_mode,
        random_seed=random_seed,
        taxonomy_lookup=taxonomy_lookup,
        when_checkpoint_path=when_checkpoint_path,
        n_jobs=n_jobs,
        collect_timing=collect_timing,
        **CHUNK_CONFIG,
    )
    val_result = precompute_chunk_cache(
        df=val_rows,
        base_dir=base_dir,
        output_dir=val_dir,
        encode_labels=encode_labels,
        sr=SR,
        duration=duration,
        chunking_mode=chunking_mode,
        random_seed=random_seed,
        taxonomy_lookup=taxonomy_lookup,
        when_checkpoint_path=when_checkpoint_path,
        n_jobs=n_jobs,
        collect_timing=collect_timing,
        **CHUNK_CONFIG,
    )

    if collect_timing:
        train_samples, train_timing = train_result
        val_samples, val_timing = val_result
        timing_stats = {
            "train": train_timing,
            "val": val_timing,
        }
    else:
        train_samples = train_result
        val_samples = val_result

    print(f"writing {train_samples_path}", flush=True)
    with open(train_samples_path, "wb") as handle:
        pickle.dump(train_samples, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"writing {val_samples_path}", flush=True)
    with open(val_samples_path, "wb") as handle:
        pickle.dump(val_samples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if collect_timing:
        return train_samples, val_samples, timing_stats
    return train_samples, val_samples
