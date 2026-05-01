from collections import defaultdict

import librosa
import numpy as np

from src.audio_processing import get_signal_centered_chunks_from_array
from src.config import CHUNK_CONFIG, DURATION, SR
from src.paths import resolve_path


N_FFT = 2048
HOP_LENGTH = 512
FMIN = 300
FMAX = 8000


def _emit_progress(message, show_progress=False, progress_hook=None):
    if progress_hook is not None:
        progress_hook(message)
    elif show_progress:
        print(message, flush=True)


def compute_mean_spectrum(
    y,
    sr=SR,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    fmin=FMIN,
    fmax=FMAX,
):
    """Return a normalized mean spectrum over the target frequency range."""
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    S_power = np.abs(S) ** 2

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    valid = (freqs >= fmin) & (freqs <= fmax)

    spectrum = np.mean(S_power[valid, :], axis=1)
    spectrum = spectrum / (spectrum.sum() + 1e-12)

    return spectrum, freqs[valid]


def build_species_templates(
    df,
    audio_col="full_path",
    label_col="primary_label",
    min_examples=5,
    max_per_species=None,
    sr=SR,
    base_dir=".",
    show_progress=False,
    progress_hook=None,
):
    """Build spectral templates from signal-centered chunks of high-quality clips."""
    species_spectra = defaultdict(list)
    shared_freqs = None

    if "rating" in df.columns:
        df = df[df["rating"] == 5]

    grouped = list(df.groupby(label_col))
    total_species = len(grouped)

    for species_idx, (species, group) in enumerate(grouped, start=1):
        if len(group) < min_examples:
            continue

        if max_per_species is not None:
            group = group.sample(n=min(len(group), max_per_species), random_state=42)

        _emit_progress(
            f"Templates: species {species_idx}/{total_species} -> {species} ({len(group)} clips)",
            show_progress=show_progress,
            progress_hook=progress_hook,
        )

        kept_for_species = 0
        for clip_idx, (_, row) in enumerate(group.iterrows(), start=1):
            try:
                audio_path = resolve_path(row[audio_col], base_dir)
                y, _ = librosa.load(audio_path, sr=sr, mono=True)

                chunk_result = get_signal_centered_chunks_from_array(
                    y=y,
                    sr=sr,
                    duration=DURATION,
                    n_fft=2048,
                    hop_length=256,
                    fmin=300,
                    fmax=None,
                    band_peak_rel_db=CHUNK_CONFIG["band_peak_rel_db"],
                    energy_smooth_frames=9,
                    background_smooth_frames=151,
                    threshold_db=CHUNK_CONFIG["threshold_db"],
                    min_event_duration=CHUNK_CONFIG["min_event_duration"],
                    merge_gap=CHUNK_CONFIG["merge_gap"],
                    fallback_to_regular_split=False,
                    only_strongest=True,
                )
                if not chunk_result["chunks"]:
                    continue

                spectrum, freqs = compute_mean_spectrum(chunk_result["chunks"][0], sr=sr)

                if shared_freqs is None:
                    shared_freqs = freqs

                species_spectra[species].append(spectrum)
                kept_for_species += 1
            except Exception:
                continue

            if show_progress and clip_idx % 25 == 0:
                _emit_progress(
                    f"  processed {clip_idx}/{len(group)} clips for {species} "
                    f"(kept {kept_for_species})",
                    show_progress=show_progress,
                    progress_hook=progress_hook,
                )

        _emit_progress(
            f"  finished {species}: kept {kept_for_species} usable clips",
            show_progress=show_progress,
            progress_hook=progress_hook,
        )

    templates = {}
    for species, specs in species_spectra.items():
        if not specs:
            continue

        template = np.mean(np.stack(specs), axis=0)
        template = template / (template.sum() + 1e-12)
        templates[species] = template

    return templates, shared_freqs


def cosine_similarity(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def chunk_similarity_to_species(
    chunk,
    species,
    templates,
    sr=SR,
):
    if species not in templates:
        return None

    chunk_spec, _ = compute_mean_spectrum(chunk, sr=sr)
    return cosine_similarity(chunk_spec, templates[species])


def filter_secondary_labels_for_chunk(
    chunk,
    secondary_labels,
    templates,
    candidate_overlap_sec=None,
    min_overlap=0.2,
    similarity_threshold=0.75,
    sr=SR,
):
    """Keep secondary labels only when timing and spectrum both support them."""
    if not secondary_labels:
        return [], {}

    if (
        candidate_overlap_sec is not None
        and candidate_overlap_sec < min_overlap
    ):
        return [], {}

    kept = []
    scores = {}

    for species in secondary_labels:
        if species not in templates:
            continue

        sim = chunk_similarity_to_species(chunk, species, templates, sr=sr)
        scores[species] = sim

        if sim is not None and sim >= similarity_threshold:
            kept.append(species)

    return kept, scores


def compute_overlap(chunk_interval, start_time, end_time):
    """Return overlap in seconds between two [start, end] intervals."""
    if chunk_interval is None or start_time is None or end_time is None:
        return 0.0

    chunk_start, chunk_end = chunk_interval
    overlap_start = max(chunk_start, start_time)
    overlap_end = min(chunk_end, end_time)

    if overlap_end <= overlap_start:
        return 0.0

    return float(overlap_end - overlap_start)
