# Audio + spectrogram config (single source of truth)

SR = 32000
DURATION = 5.0

N_FFT = 1024
HOP_LENGTH = 320
N_MELS = 128
FMIN = 20
FMAX = 14000

CHUNK_CONFIG = {
    "threshold_db": 6.0,
    "band_peak_rel_db": 6.0,
    "min_event_duration": 0.18,
    "merge_gap": 0.8,
}