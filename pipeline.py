import json
import argparse
import pickle
from pathlib import Path
import numpy as np

def build_encoder(label_to_idx):
    num_classes = len(label_to_idx)

    def encode(labels):
        target = np.zeros(num_classes, dtype=np.float32)
        for label in labels:
            if label in label_to_idx:
                target[label_to_idx[label]] = 1.0
        return target

    return encode


def load_templates(templates_path):
    path = Path(templates_path)
    if not path.exists():
        return None

    with open(path, "rb") as handle:
        payload = pickle.load(handle)

    if isinstance(payload, dict) and "templates" in payload:
        return payload["templates"]
    return payload


def run_prepare(args):
    from src.metadata import prepare_metadata_artifacts

    prepare_metadata_artifacts(
        base_dir=args.base_dir,
        output_dir=args.splits_dir,
        test_size=args.test_size,
        n_splits=args.n_splits,
        random_state=args.random_state,
        build_templates_flag=not args.skip_templates,
        templates_path=args.templates_path,
    )


def run_precompute(args):
    from src.metadata import load_metadata_artifacts
    from src.precompute import precompute_fold

    metadata = load_metadata_artifacts(args.base_dir, input_dir=args.splits_dir)
    encode_labels = build_encoder(metadata["label_to_idx"])
    templates = load_templates(args.templates_path) if args.templates_path else None

    if args.all_folds:
        folds = sorted(
            set(
                metadata["file_level_df"]
                .loc[metadata["file_level_df"]["cv_fold"] >= 0, "cv_fold"]
                .tolist()
            )
        )
    else:
        folds = [args.fold]

    for fold in folds:
        print(f"precomputing fold {fold}")
        precompute_fold(
            fold=fold,
            file_level_df=metadata["file_level_df"],
            master_df=metadata["master_df"],
            templates=templates,
            encode_labels=encode_labels,
            taxonomy_lookup=metadata.get("taxonomy_lookup"),
            base_dir=args.base_dir,
            base_output_dir=args.data_processed_dir,
            n_jobs=args.n_jobs,
            force=args.force,
            chunking_mode=args.chunking_mode,
            random_seed=args.random_seed,
            when_checkpoint_path=args.when_checkpoint,
        )


def run_train_when(args):
    from src.train_when import train_when_detector

    result = train_when_detector(
        fold=args.fold,
        base_dir=args.base_dir,
        splits_dir=args.splits_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        device=args.device,
    )
    print(json.dumps(result, indent=2))


def run_train(args):
    from src.metadata import load_metadata_artifacts
    from src.train import parse_aug_config_arg, parse_curriculum_arg, train_fold

    aug_params = parse_aug_config_arg(args.aug_config)
    curriculum_stages = parse_curriculum_arg(args.curriculum_config)

    if args.all_folds:
        metadata = load_metadata_artifacts(args.base_dir, input_dir=args.splits_dir)
        folds = sorted(
            set(
                metadata["file_level_df"]
                .loc[metadata["file_level_df"]["cv_fold"] >= 0, "cv_fold"]
                .tolist()
            )
        )
    else:
        folds = [args.fold]

    for fold in folds:
        print(f"training fold {fold}")
        train_fold(
            fold=fold,
            base_dir=args.base_dir,
            data_processed_dir=args.data_processed_dir,
            splits_dir=args.splits_dir,
            output_dir=args.checkpoints_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            num_workers=args.num_workers,
            augment=args.augment,
            curriculum=args.curriculum,
            curriculum_stages=curriculum_stages,
            aug_params=aug_params,
            debug_errors=args.debug_errors,
            debug_decision_threshold=args.debug_decision_threshold,
            debug_low_conf_threshold=args.debug_low_conf_threshold,
            debug_top_k=args.debug_top_k,
            dropout=args.dropout,
            early_stopping_patience=args.early_stopping_patience,
            secondary_label_total_weight=args.secondary_label_total_weight,
            device=args.device,
        )


def run_infer(args):
    from src.inference import predict_file

    result = predict_file(
        audio_path=args.audio_path,
        checkpoint_path=args.checkpoint,
        top_k=args.top_k,
        device=args.device,
        base_dir=args.base_dir,
        inference_mode=args.inference_mode,
        window_duration=args.window_duration,
        window_hop=args.window_hop,
        aggregation=args.aggregation,
    )
    print(json.dumps(result, indent=2))


def run_debug_precompute(args):
    from src.debug_subset import run_debug_precompute
    from src.metadata import load_metadata_artifacts

    metadata = load_metadata_artifacts(args.base_dir, input_dir=args.splits_dir)
    templates = load_templates(args.templates_path) if args.templates_path else None

    phases = None
    if args.phases:
        phases = [phase.strip() for phase in args.phases.split(",") if phase.strip()]

    result = run_debug_precompute(
        master_df=metadata["master_df"],
        file_level_df=metadata["file_level_df"],
        label_to_idx=metadata["label_to_idx"],
        templates=templates,
        base_dir=args.base_dir,
        output_dir=args.output_dir,
        fold=args.fold,
        samples_per_phase=args.samples_per_phase,
        phases=phases,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        chunking_mode=args.chunking_mode,
        random_seed=args.random_seed,
    )
    print(json.dumps(result, indent=2))


def run_debug_inspect(args):
    from src.debug_inspect import write_debug_inspection

    payload, output_path = write_debug_inspection(
        debug_dir=args.debug_dir,
        output_path=args.output_path,
        base_dir=args.base_dir,
    )
    print(json.dumps(
        {
            "summary": payload["summary"],
            "output_path": str(output_path),
        },
        indent=2,
    ))


def run_compare_runs(args):
    from src.compare_runs import write_training_comparison

    with open(args.spec_path, "r", encoding="utf-8") as handle:
        run_specs = json.load(handle)

    payload, output_path = write_training_comparison(
        run_specs=run_specs,
        output_path=args.output_path,
    )
    print(json.dumps(
        {
            "best_run": payload["best_run"],
            "output_path": str(output_path),
        },
        indent=2,
    ))


def main():
    parser = argparse.ArgumentParser(description="BirdCLEF end-to-end pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--base-dir", default=".")
    prepare_parser.add_argument("--splits-dir", default="splits")
    prepare_parser.add_argument("--templates-path", default="templates.pkl")
    prepare_parser.add_argument("--test-size", type=float, default=0.15)
    prepare_parser.add_argument("--n-splits", type=int, default=5)
    prepare_parser.add_argument("--random-state", type=int, default=42)
    prepare_parser.add_argument("--skip-templates", action="store_true")
    prepare_parser.set_defaults(func=run_prepare)

    precompute_parser = subparsers.add_parser("precompute")
    precompute_parser.add_argument("--base-dir", default=".")
    precompute_parser.add_argument("--splits-dir", default="splits")
    precompute_parser.add_argument("--data-processed-dir", default="data_processed")
    precompute_parser.add_argument("--templates-path", default="templates.pkl")
    precompute_parser.add_argument("--fold", type=int, default=0)
    precompute_parser.add_argument("--all-folds", action="store_true")
    precompute_parser.add_argument("--n-jobs", type=int, default=3)
    precompute_parser.add_argument("--force", action="store_true")
    precompute_parser.add_argument("--chunking-mode", choices=["signal", "sequential", "random", "hybrid", "when"], default="signal")
    precompute_parser.add_argument("--random-seed", type=int, default=42)
    precompute_parser.add_argument("--when-checkpoint", default=None)
    precompute_parser.set_defaults(func=run_precompute)

    train_when_parser = subparsers.add_parser("train-when")
    train_when_parser.add_argument("--base-dir", default=".")
    train_when_parser.add_argument("--splits-dir", default="splits")
    train_when_parser.add_argument("--fold", type=int, default=0)
    train_when_parser.add_argument("--output-dir", default="when_checkpoints")
    train_when_parser.add_argument("--epochs", type=int, default=8)
    train_when_parser.add_argument("--batch-size", type=int, default=16)
    train_when_parser.add_argument("--lr", type=float, default=1e-3)
    train_when_parser.add_argument("--num-workers", type=int, default=0)
    train_when_parser.add_argument("--device", default=None)
    train_when_parser.set_defaults(func=run_train_when)

    debug_precompute_parser = subparsers.add_parser("debug-precompute")
    debug_precompute_parser.add_argument("--base-dir", default=".")
    debug_precompute_parser.add_argument("--splits-dir", default="splits")
    debug_precompute_parser.add_argument("--templates-path", default="templates.pkl")
    debug_precompute_parser.add_argument("--output-dir", default="debug_precompute")
    debug_precompute_parser.add_argument("--fold", type=int, default=0)
    debug_precompute_parser.add_argument("--samples-per-phase", type=int, default=5)
    debug_precompute_parser.add_argument("--phases", default="clean,semi,soundscape,noisy")
    debug_precompute_parser.add_argument("--random-state", type=int, default=42)
    debug_precompute_parser.add_argument("--n-jobs", type=int, default=1)
    debug_precompute_parser.add_argument("--chunking-mode", choices=["signal", "sequential", "random", "hybrid"], default="signal")
    debug_precompute_parser.add_argument("--random-seed", type=int, default=42)
    debug_precompute_parser.set_defaults(func=run_debug_precompute)

    debug_inspect_parser = subparsers.add_parser("debug-inspect")
    debug_inspect_parser.add_argument("--base-dir", default=".")
    debug_inspect_parser.add_argument("--debug-dir", default="debug_precompute")
    debug_inspect_parser.add_argument("--output-path", default=None)
    debug_inspect_parser.set_defaults(func=run_debug_inspect)

    compare_runs_parser = subparsers.add_parser("compare-runs")
    compare_runs_parser.add_argument("spec_path")
    compare_runs_parser.add_argument("--output-path", default="run_comparison.json")
    compare_runs_parser.set_defaults(func=run_compare_runs)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--base-dir", default=".")
    train_parser.add_argument("--splits-dir", default="splits")
    train_parser.add_argument("--data-processed-dir", default="data_processed")
    train_parser.add_argument("--checkpoints-dir", default="checkpoints")
    train_parser.add_argument("--fold", type=int, default=0)
    train_parser.add_argument("--all-folds", action="store_true")
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--num-workers", type=int, default=0)
    train_parser.add_argument("--augment", action="store_true")
    train_parser.add_argument("--aug-config", default=None)
    train_parser.add_argument("--curriculum", action="store_true")
    train_parser.add_argument("--curriculum-config", default=None)
    train_parser.add_argument("--debug-errors", action="store_true")
    train_parser.add_argument("--debug-decision-threshold", type=float, default=0.5)
    train_parser.add_argument("--debug-low-conf-threshold", type=float, default=0.6)
    train_parser.add_argument("--debug-top-k", type=int, default=5)
    train_parser.add_argument("--dropout", type=float, default=0.2)
    train_parser.add_argument("--early-stopping-patience", type=int, default=None)
    train_parser.add_argument("--secondary-label-total-weight", type=float, default=None)
    train_parser.add_argument("--device", default=None)
    train_parser.set_defaults(func=run_train)

    infer_parser = subparsers.add_parser("infer")
    infer_parser.add_argument("audio_path")
    infer_parser.add_argument("--checkpoint", required=True)
    infer_parser.add_argument("--top-k", type=int, default=10)
    infer_parser.add_argument("--device", default=None)
    infer_parser.add_argument("--base-dir", default=".")
    infer_parser.add_argument("--inference-mode", choices=["auto", "clip", "soundscape"], default="auto")
    infer_parser.add_argument("--window-duration", type=float, default=5.0)
    infer_parser.add_argument("--window-hop", type=float, default=5.0)
    infer_parser.add_argument("--aggregation", choices=["max", "mean", "meanmax"], default="max")
    infer_parser.set_defaults(func=run_infer)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
