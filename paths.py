from pathlib import Path


KNOWN_ROOTS = [
    "train_audio",
    "train_soundscapes",
    "data_processed",
    "debug_precompute",
    "debug_precompute_smoke",
    "debug_precompute_smoke_v2",
    "splits",
    "splits_smoke",
    "splits_smoke_v2",
    "checkpoints",
]


def normalize_base_dir(base_dir):
    return Path(base_dir).expanduser().resolve()


def relativize_to_base(path_value, base_dir):
    if path_value is None:
        return None

    base_path = normalize_base_dir(base_dir)
    path = Path(path_value)
    if not path.is_absolute():
        return str(path)

    try:
        return str(path.resolve().relative_to(base_path))
    except Exception:
        return str(path)


def resolve_path(path_value, base_dir="."):
    if path_value is None:
        return None

    path = Path(path_value)
    if path.exists():
        return str(path)

    base_path = normalize_base_dir(base_dir)

    if not path.is_absolute():
        candidate = base_path / path
        if candidate.exists():
            return str(candidate)

    parts = list(path.parts)
    for root_name in KNOWN_ROOTS:
        if root_name in parts:
            idx = parts.index(root_name)
            candidate = base_path.joinpath(*parts[idx:])
            if candidate.exists():
                return str(candidate)

    return str(base_path / path) if not path.is_absolute() else str(path)
