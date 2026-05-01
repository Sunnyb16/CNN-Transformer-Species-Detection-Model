import numpy as np

TAXONOMY_PREFIX = "taxon:"


def clean_secondary_labels(df):
    df = df.copy()
    df["secondary_labels"] = df["secondary_labels"].apply(
        lambda x: x if isinstance(x, list) else []
    )
    return df


def build_all_labels(df):
    df = df.copy()
    df["all_labels"] = df.apply(
        lambda row: sorted(set([row["primary_label"]] + row["secondary_labels"])),
        axis=1
    )
    return df


def build_class_list(df):
    species_class_list = sorted({
        label
        for labels in df["all_labels"]
        for label in labels
    })
    taxonomy_class_list = sorted({
        label
        for labels in df.get("taxonomy_labels", [])
        for label in labels
    })
    class_list = species_class_list + taxonomy_class_list
    label_to_idx = {label: i for i, label in enumerate(class_list)}
    return class_list, label_to_idx, species_class_list, taxonomy_class_list


def taxonomy_token(class_name):
    return f"{TAXONOMY_PREFIX}{class_name}"


def labels_to_taxonomy_labels(labels, taxonomy_lookup):
    taxonomy_labels = []
    seen = set()

    for label in labels:
        class_name = taxonomy_lookup.get(str(label))
        if not class_name:
            continue

        token = taxonomy_token(class_name)
        if token in seen:
            continue

        seen.add(token)
        taxonomy_labels.append(token)

    return taxonomy_labels


def encode_labels(label_set, label_to_idx, num_classes):
    target = np.zeros(num_classes, dtype=np.float32)
    for l in label_set:
        target[label_to_idx[l]] = 1.0
    return target
