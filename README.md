src Package
This folder contains the BirdCLEF audio classification pipeline. It prepares metadata and folds, precomputes log-mel spectrogram chunks, trains a multi-label ResNet model, runs inference on clips or soundscapes, and provides debugging utilities for inspecting preprocessing and training runs.

Run commands from the repository root so imports like src.pipeline resolve correctly.

Setup
Install the project dependencies:

pip install -r requirements.txt
The pipeline expects BirdCLEF-style data under the project base directory:

train.csv
taxonomy.csv
train_audio/
train_soundscapes/
train_soundscapes_labels.csv
Generated artifacts are written to folders such as splits/, data_processed/, checkpoints/, when_checkpoints/, and debug_precompute/.

Typical Workflow
1. Prepare metadata and folds
Build the master metadata tables, label maps, train/validation folds, and optional species templates:

python -m src.pipeline prepare \
  --base-dir . \
  --splits-dir splits \
  --templates-path templates.pkl \
  --test-size 0.15 \
  --n-splits 5
Use --skip-templates if you do not want to build template filters.

2. Precompute spectrogram chunks
Create cached .npy log-mel spectrogram chunks for one fold:

python -m src.pipeline precompute \
  --base-dir . \
  --splits-dir splits \
  --data-processed-dir data_processed \
  --templates-path templates.pkl \
  --fold 0 \
  --n-jobs 3 \
  --chunking-mode signal
Precompute every available fold:

python -m src.pipeline precompute --all-folds --base-dir .
Supported chunking modes are:

signal: center chunks around detected signal intervals.
sequential: split audio into regular fixed-length windows.
random: sample random fixed-length chunks.
when: use a trained WHEN detector to center chunks.
hybrid: choose the chunking strategy from each row's phase group.
3. Train the classifier
Train one fold:

python -m src.pipeline train \
  --base-dir . \
  --splits-dir splits \
  --data-processed-dir data_processed \
  --checkpoints-dir checkpoints \
  --fold 0 \
  --epochs 10 \
  --batch-size 32 \
  --augment
Train all folds:

python -m src.pipeline train --all-folds --base-dir .
Useful training options:

--curriculum: train in phase stages from clean data toward noisier data.
--aug-config path-or-json: override augmentation probabilities and ranges.
--debug-errors: write validation false positive, false negative, and low-confidence reports.
--early-stopping-patience N: stop after N epochs without improvement.
--secondary-label-total-weight W: downweight secondary labels while keeping the primary label at 1.0.
4. Run inference
Predict the top labels for one audio file:

python -m src.pipeline infer path/to/audio.ogg \
  --checkpoint checkpoints/fold_0_best.pt \
  --top-k 10 \
  --inference-mode auto \
  --aggregation max
Inference modes:

auto: use soundscape windows for long files or soundscape-like names; otherwise use clip chunking.
clip: use signal-centered chunking for a single clip.
soundscape: use sliding fixed-length soundscape windows.
Aggregation options are max, mean, and meanmax.

WHEN Detector
The WHEN detector is a small auxiliary model that learns frame-level event timing from clean clips. It can be used by the precompute step with --chunking-mode when or --chunking-mode hybrid.

Train a WHEN detector:

python -m src.pipeline train-when \
  --base-dir . \
  --splits-dir splits \
  --fold 0 \
  --output-dir when_checkpoints \
  --epochs 8 \
  --batch-size 16
Use it during precompute:

python -m src.pipeline precompute \
  --base-dir . \
  --fold 0 \
  --chunking-mode when \
  --when-checkpoint when_checkpoints/when_fold_0_best.pt
Debugging Commands
Build a small debug precompute dataset:

python -m src.pipeline debug-precompute \
  --base-dir . \
  --splits-dir splits \
  --output-dir debug_precompute \
  --fold 0 \
  --samples-per-phase 5
Inspect debug precompute outputs:

python -m src.pipeline debug-inspect \
  --base-dir . \
  --debug-dir debug_precompute
Compare training runs from a JSON spec:

python -m src.pipeline compare-runs run_specs.json \
  --output-path run_comparison.json
Example run_specs.json:

[
  {
    "name": "baseline_fold_0",
    "checkpoint": "checkpoints/fold_0_best.pt",
    "debug_report": "checkpoints/fold_0_debug_report.json"
  }
]
Module Map
pipeline.py: command-line entry point for prepare, precompute, train, inference, and debug tasks.
metadata.py: loads BirdCLEF CSVs, builds master/file-level metadata, labels, taxonomy targets, and fold artifacts.
splits.py: creates holdout and cross-validation splits at the file level.
precompute.py: extracts audio chunks, converts them to log-mel spectrograms, and writes fold caches.
audio_processing.py: signal detection, chunking, padding/cropping, and spectrogram helpers.
dataset.py: PyTorch dataset for cached spectrograms and optional audio/spec augmentation.
model.py: ResNet18-based multi-label classifier for log-mel inputs.
train.py: classifier training loop, curriculum learning, validation metrics, checkpoints, and error reports.
inference.py: checkpoint loading and top-k prediction for clips or soundscapes.
when_model.py, when_detector.py, train_when.py: auxiliary event-timing model and chunk selection helpers.
templates.py: species template creation and secondary-label filtering.
eval_debug.py, debug_subset.py, debug_inspect.py, compare_runs.py: diagnostics and run comparison utilities.
config.py: shared sample rate, duration, mel, FFT, and signal chunking settings.
paths.py: path resolution helpers for project-relative artifacts.
labels.py: label cleaning, taxonomy labels, and target encoding helpers.
Core Defaults
Audio and spectrogram defaults are defined in config.py:

sample rate: 32000 Hz
chunk duration: 5.0 seconds
n_fft: 1024
hop_length: 320
n_mels: 128
frequency range: 20-14000 Hz
Signal-centered chunking uses the shared CHUNK_CONFIG thresholds from config.py.

Checkpoint Contents
Classifier checkpoints contain:

class_list: full model output classes, including taxonomy classes.
species_class_list: species-only labels used for inference ranking.
model_state_dict: trained PyTorch weights.
history: per-epoch training and validation metrics.
best_val_loss, best_val_auc, and best_epoch.
Training options such as dropout and secondary_label_total_weight.
Use checkpoints created by src.train or src.pipeline train; older checkpoints without class_list and model_state_dict will not load in inference.py.
