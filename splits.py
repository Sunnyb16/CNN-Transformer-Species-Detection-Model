import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold


# -----------------------------
# Phase selection
# -----------------------------
phase_priority = {
    "clean": 0,
    "semi": 1,
    "noisy": 2,
    "soundscape": 3,
}

def choose_phase(series):
    vals = [x for x in series.dropna() if x in phase_priority]
    if not vals:
        return np.nan
    return max(vals, key=lambda x: phase_priority[x])


# -----------------------------
# Merge labels across rows
# -----------------------------
def merge_label_lists(series):
    labels = set()
    for val in series:
        if isinstance(val, list):
            labels.update(val)
    return sorted(labels)


# -----------------------------
# Build file-level dataframe
# -----------------------------
def build_file_level_df(master_df):
    grouped = master_df.groupby("file_id")

    file_level_df = grouped.agg({
        "full_path": "first",
        "filename": "first",
        "rating": "first",
    }).reset_index()

    file_level_df["phase_group"] = grouped["phase_group"].apply(choose_phase).values
    file_level_df["all_labels"] = grouped["all_labels"].apply(merge_label_lists).values

    return file_level_df



# Create train/test + CV splits
def create_splits(file_level_df, test_size=0.15, n_splits=5, random_state=42):

    file_level_df = file_level_df.copy()

    # -----------------------------
    # 1. Holdout split
    # -----------------------------
    trainval_df, holdout_df = train_test_split(
        file_level_df,
        test_size=test_size,
        random_state=random_state,
        stratify=file_level_df["phase_group"]
    )

    trainval_ids = set(trainval_df["file_id"])
    holdout_ids = set(holdout_df["file_id"])

    file_level_df["split_role"] = file_level_df["file_id"].apply(
        lambda x: "holdout_test" if x in holdout_ids else "trainval"
    )

    # -----------------------------
    # 2. CV folds
    # -----------------------------
    trainval_unique = file_level_df[file_level_df["split_role"] == "trainval"].copy()

    print("DEBUG: trainval size =", len(trainval_unique))

    if len(trainval_unique) == 0:
        raise ValueError("No trainval data — split failed")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    trainval_unique["cv_fold"] = -1

    for fold, (_, val_idx) in enumerate(
        skf.split(trainval_unique["file_id"], trainval_unique["phase_group"])
    ):
        trainval_unique.iloc[val_idx, trainval_unique.columns.get_loc("cv_fold")] = fold

    print("DEBUG: cv_fold distribution:")
    print(trainval_unique["cv_fold"].value_counts())

    # -----------------------------
    # 3. Merge back
    # -----------------------------
    # Remove old cv_fold if it exists
    if "cv_fold" in file_level_df.columns:
        file_level_df = file_level_df.drop(columns=["cv_fold"])

    # Merge new cv_fold
    file_level_df = file_level_df.merge(
        trainval_unique[["file_id", "cv_fold"]],
        on="file_id",
        how="left"
    )

    # Fill missing (holdout)
    file_level_df["cv_fold"] = file_level_df["cv_fold"].fillna(-1).astype(int)

    return file_level_df
