import json
import pickle
from pathlib import Path

import pandas as pd

import src.precompute as pc
from src.config import CHUNK_CONFIG, DURATION, SR
from src.paths import relativize_to_base, resolve_path


DEFAULT_DEBUG_PHASES = ["clean", "semi", "soundscape", "noisy"]


def build_encoder(label_to_idx):
    num_classes = len(label_to_idx)

    def encode(labels):
        target = [0.0] * num_classes
        for label in labels:
            if label in label_to_idx:
                target[label_to_idx[label]] = 1.0
        return target

    return encode


def sample_debug_rows(
    master_df,
    file_level_df,
    fold=0,
    samples_per_phase=5,
    phases=None,
    random_state=42,
):
    phases = phases or DEFAULT_DEBUG_PHASES

    split_cols = file_level_df[["file_id", "split_role", "cv_fold"]].copy()
    merged = master_df.merge(split_cols, on="file_id", how="left")

    candidate_rows = merged[
        (merged["split_role"] == "trainval") & (merged["cv_fold"] != fold)
    ].copy()

    samples = []
    summary = {}

    for phase in phases:
        phase_df = candidate_rows[candidate_rows["phase_group"] == phase].copy()
        summary[phase] = int(len(phase_df))

        if phase_df.empty:
            continue

        take_n = min(samples_per_phase, len(phase_df))
        sampled = phase_df.sample(n=take_n, random_state=random_state)
        sampled = sampled.assign(debug_selected_phase=phase)
        samples.append(sampled)

    if not samples:
        raise RuntimeError("No rows were selected for the debug subset.")

    debug_df = pd.concat(samples, ignore_index=True)

    return debug_df, summary


def run_debug_precompute(
    master_df,
    file_level_df,
    label_to_idx,
    templates,
    base_dir,
    output_dir,
    fold=0,
    samples_per_phase=5,
    phases=None,
    random_state=42,
    n_jobs=1,
    chunking_mode="signal",
    random_seed=42,
):
    debug_rows, summary = sample_debug_rows(
        master_df=master_df,
        file_level_df=file_level_df,
        fold=fold,
        samples_per_phase=samples_per_phase,
        phases=phases,
        random_state=random_state,
    )

    output_path = Path(resolve_path(output_dir, base_dir))
    output_path.mkdir(parents=True, exist_ok=True)

    manifest_path = output_path / "debug_rows.csv"
    debug_rows.to_csv(manifest_path, index=False)

    encode_labels = build_encoder(label_to_idx)
    pc.TEMPLATES = templates
    samples = pc.precompute_chunk_cache(
        df=debug_rows,
        base_dir=base_dir,
        output_dir=str(output_path / "specs"),
        encode_labels=encode_labels,
        n_jobs=n_jobs,
        sr=SR,
        duration=DURATION,
        threshold_db=CHUNK_CONFIG["threshold_db"],
        band_peak_rel_db=CHUNK_CONFIG["band_peak_rel_db"],
        min_event_duration=CHUNK_CONFIG["min_event_duration"],
        merge_gap=CHUNK_CONFIG["merge_gap"],
        only_strongest=False,
        fallback_to_regular_split=True,
        chunking_mode=chunking_mode,
        random_seed=random_seed,
    )

    samples_path = output_path / "debug_samples.pkl"
    with open(samples_path, "wb") as handle:
        pickle.dump(samples, handle)

    phase_sample_counts = {}
    for sample in samples:
        phase = sample.get("phase_group", "unknown")
        phase_sample_counts[phase] = phase_sample_counts.get(phase, 0) + 1

    summary_payload = {
        "fold": fold,
        "samples_per_phase_requested": samples_per_phase,
        "phases": phases or DEFAULT_DEBUG_PHASES,
        "candidate_rows_per_phase": summary,
        "selected_rows": int(len(debug_rows)),
        "generated_samples": int(len(samples)),
        "generated_samples_per_phase": phase_sample_counts,
        "chunking_mode": chunking_mode,
        "random_seed": random_seed,
        "manifest_path": str(manifest_path),
        "samples_path": str(samples_path),
    }

    with open(output_path / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)

    return summary_payload
