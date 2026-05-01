import ast
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.labels import (
    build_class_list,
    build_all_labels,
    clean_secondary_labels,
    labels_to_taxonomy_labels,
)
from src.paths import resolve_path
from src.splits import build_file_level_df, create_splits


PHASE_POLICY = {
    "clean_min_rating": 4.5,
    "semi_min_rating": 3.0,
    "clean_max_secondary": 1,
    "semi_max_secondary": 3,
}


def _emit_progress(show_progress, message, progress_hook=None):
    if progress_hook is not None:
        progress_hook(message)
    elif show_progress:
        print(message, flush=True)


def parse_secondary_labels(value):
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []
    return []


def parse_soundscape_labels(value):
    if not isinstance(value, str) or not value.strip():
        return []
    return [label.strip() for label in value.split(";") if label.strip()]


def time_to_seconds(value):
    if pd.isna(value):
        return np.nan
    parts = [float(part) for part in str(value).split(":")]
    if len(parts) != 3:
        raise ValueError(f"Unsupported time format: {value}")
    hours, minutes, seconds = parts
    return (hours * 3600.0) + (minutes * 60.0) + seconds


def build_train_clip_metadata(base_dir):
    train_path = Path(base_dir) / "train.csv"
    df = pd.read_csv(train_path)

    df["secondary_labels"] = df["secondary_labels"].apply(parse_secondary_labels)
    df = clean_secondary_labels(df)
    df = build_all_labels(df)

    filenames = df["filename"].map(lambda value: Path(value).name)
    df["full_path"] = (
        "train_audio/"
        + df["primary_label"].astype(str)
        + "/"
        + filenames.astype(str)
    )
    df["file_id"] = df["filename"].apply(lambda value: Path(value).stem)
    df["start_time"] = np.nan
    df["end_time"] = np.nan
    df["is_soundscape"] = False

    rating = df["rating"].fillna(-1.0)
    secondary_count = df["secondary_labels"].apply(len)

    clean_mask = (
        (rating >= PHASE_POLICY["clean_min_rating"])
        & (secondary_count <= PHASE_POLICY["clean_max_secondary"])
    )
    semi_mask = (
        (rating >= PHASE_POLICY["semi_min_rating"])
        & ~clean_mask
        & (secondary_count <= PHASE_POLICY["semi_max_secondary"])
    )

    df["phase_group"] = np.select(
        [
            clean_mask,
            semi_mask,
        ],
        [
            "clean",
            "semi",
        ],
        default="noisy",
    )

    return df


def build_soundscape_metadata(base_dir):
    labels_path = Path(base_dir) / "train_soundscapes_labels.csv"
    df = pd.read_csv(labels_path)

    df["label_list"] = df["primary_label"].apply(parse_soundscape_labels)
    df = df[df["label_list"].map(len) > 0].copy()

    df["full_path"] = "train_soundscapes/" + df["filename"].astype(str)
    df["file_id"] = df["filename"].apply(lambda value: Path(value).stem)
    df["start_time"] = df["start"].apply(time_to_seconds)
    df["end_time"] = df["end"].apply(time_to_seconds)
    df["primary_label"] = df["label_list"].apply(lambda labels: labels[0])
    df["secondary_labels"] = df["label_list"].apply(lambda labels: labels[1:])
    df["all_labels"] = df["label_list"]
    df["rating"] = np.nan
    df["phase_group"] = "soundscape"
    df["is_soundscape"] = True

    return df[
        [
            "primary_label",
            "secondary_labels",
            "rating",
            "filename",
            "full_path",
            "file_id",
            "start_time",
            "end_time",
            "all_labels",
            "phase_group",
            "is_soundscape",
        ]
    ].copy()


def build_master_dataframe(base_dir):
    clips_df = build_train_clip_metadata(base_dir)
    soundscape_df = build_soundscape_metadata(base_dir)

    common_columns = sorted(set(clips_df.columns) | set(soundscape_df.columns))
    clips_df = clips_df.reindex(columns=common_columns)
    soundscape_df = soundscape_df.reindex(columns=common_columns)

    master_df = pd.concat([clips_df, soundscape_df], ignore_index=True)
    master_df["path_exists"] = [
        os.path.exists(resolve_path(value, base_dir))
        for value in master_df["full_path"]
    ]
    master_df = master_df[master_df["path_exists"]].copy()

    return master_df


def load_taxonomy_lookup(base_dir):
    taxonomy_path = Path(base_dir) / "taxonomy.csv"
    if not taxonomy_path.exists():
        return {}

    taxonomy_df = pd.read_csv(taxonomy_path, dtype={"primary_label": str})
    taxonomy_df["primary_label"] = taxonomy_df["primary_label"].astype(str)
    taxonomy_df["class_name"] = taxonomy_df["class_name"].astype(str)

    return {
        row["primary_label"]: row["class_name"]
        for _, row in taxonomy_df.iterrows()
        if isinstance(row["class_name"], str) and row["class_name"]
    }


def attach_targets(df, label_to_idx, taxonomy_lookup):
    num_classes = len(label_to_idx)

    def encode(labels):
        target = np.zeros(num_classes, dtype=np.float32)
        for label in labels:
            if label in label_to_idx:
                target[label_to_idx[label]] = 1.0
        return target

    df = df.copy()
    df["taxonomy_labels"] = df["all_labels"].apply(
        lambda labels: labels_to_taxonomy_labels(labels, taxonomy_lookup)
    )
    df["model_labels"] = df.apply(
        lambda row: row["all_labels"] + row["taxonomy_labels"],
        axis=1,
    )
    df["target"] = df["model_labels"].apply(encode)
    return df


def save_dataframe(df, path_base, save_csv=True):
    df.to_pickle(f"{path_base}.pkl")
    if not save_csv:
        return

    csv_df = df.copy()
    for col in csv_df.columns:
        csv_df[col] = csv_df[col].apply(
            lambda value: json.dumps(value.tolist())
            if isinstance(value, np.ndarray)
            else json.dumps(value)
            if isinstance(value, list)
            else value
        )
    csv_df.to_csv(f"{path_base}.csv", index=False)


def prepare_metadata_artifacts(
    base_dir,
    output_dir="splits",
    test_size=0.15,
    n_splits=5,
    random_state=42,
    build_templates_flag=True,
    templates_path=None,
    save_csv=True,
    show_progress=False,
    progress_hook=None,
):
    output_path = Path(base_dir) / output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    _emit_progress(show_progress, "Loading clip and soundscape metadata...", progress_hook)
    master_df = build_master_dataframe(base_dir)
    taxonomy_lookup = load_taxonomy_lookup(base_dir)
    _emit_progress(
        show_progress,
        f"Loaded {len(master_df):,} metadata rows. Building class map...",
        progress_hook,
    )
    master_df = attach_targets(master_df, {}, taxonomy_lookup)
    class_list, label_to_idx, species_class_list, taxonomy_class_list = build_class_list(master_df)
    master_df = attach_targets(master_df, label_to_idx, taxonomy_lookup)

    _emit_progress(show_progress, "Building file-level split dataframe...", progress_hook)
    file_level_df = build_file_level_df(master_df)
    file_level_df = attach_targets(file_level_df, label_to_idx, taxonomy_lookup)
    _emit_progress(show_progress, "Creating train/validation folds...", progress_hook)
    file_level_df = create_splits(
        file_level_df,
        test_size=test_size,
        n_splits=n_splits,
        random_state=random_state,
    )

    _emit_progress(show_progress, "Saving metadata artifacts...", progress_hook)
    save_dataframe(master_df, str(output_path / "master_df"), save_csv=save_csv)
    save_dataframe(file_level_df, str(output_path / "file_level_df"), save_csv=save_csv)

    with open(output_path / "class_list.json", "w", encoding="utf-8") as handle:
        json.dump(class_list, handle, indent=2)

    with open(output_path / "species_class_list.json", "w", encoding="utf-8") as handle:
        json.dump(species_class_list, handle, indent=2)

    with open(output_path / "taxonomy_class_list.json", "w", encoding="utf-8") as handle:
        json.dump(taxonomy_class_list, handle, indent=2)

    with open(output_path / "label_to_idx.json", "w", encoding="utf-8") as handle:
        json.dump(label_to_idx, handle, indent=2, sort_keys=True)

    if templates_path is None:
        templates_path = Path(base_dir) / "templates.pkl"
    else:
        templates_path = Path(templates_path)
        if not templates_path.is_absolute():
            templates_path = Path(base_dir) / templates_path

    if build_templates_flag:
        _emit_progress(show_progress, "Building species templates...", progress_hook)
        from src.templates import build_species_templates

        template_rows = master_df[
            (~master_df["is_soundscape"]) & (master_df["rating"] == 5)
        ].copy()
        templates, shared_freqs = build_species_templates(
            template_rows,
            base_dir=base_dir,
            show_progress=show_progress,
            progress_hook=progress_hook,
        )
        with open(templates_path, "wb") as handle:
            pickle.dump(
                {
                    "templates": templates,
                    "freqs": shared_freqs,
                },
                handle,
            )
        _emit_progress(show_progress, f"Saved templates to {templates_path}", progress_hook)
    else:
        _emit_progress(show_progress, "Skipping template build.", progress_hook)

    _emit_progress(show_progress, f"Finished writing metadata to {output_path}", progress_hook)

    return {
        "master_df": master_df,
        "file_level_df": file_level_df,
        "class_list": class_list,
        "species_class_list": species_class_list,
        "taxonomy_class_list": taxonomy_class_list,
        "label_to_idx": label_to_idx,
        "taxonomy_lookup": taxonomy_lookup,
        "templates_path": str(templates_path),
    }


def load_metadata_artifacts(base_dir, input_dir="splits"):
    input_path = Path(base_dir) / input_dir

    master_df = pd.read_pickle(input_path / "master_df.pkl")
    file_level_df = pd.read_pickle(input_path / "file_level_df.pkl")

    with open(input_path / "class_list.json", "r", encoding="utf-8") as handle:
        class_list = json.load(handle)

    species_class_list_path = input_path / "species_class_list.json"
    if species_class_list_path.exists():
        with open(species_class_list_path, "r", encoding="utf-8") as handle:
            species_class_list = json.load(handle)
    else:
        species_class_list = class_list

    taxonomy_class_list_path = input_path / "taxonomy_class_list.json"
    if taxonomy_class_list_path.exists():
        with open(taxonomy_class_list_path, "r", encoding="utf-8") as handle:
            taxonomy_class_list = json.load(handle)
    else:
        taxonomy_class_list = []

    with open(input_path / "label_to_idx.json", "r", encoding="utf-8") as handle:
        label_to_idx = json.load(handle)

    return {
        "master_df": master_df,
        "file_level_df": file_level_df,
        "class_list": class_list,
        "species_class_list": species_class_list,
        "taxonomy_class_list": taxonomy_class_list,
        "label_to_idx": label_to_idx,
        "taxonomy_lookup": load_taxonomy_lookup(base_dir),
    }
