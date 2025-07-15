#!/usr/bin/env bash
set -euo pipefail

# ================================
# SHuBERT Feature Extraction + Inference Pipeline
# ================================

# --------- Configuration ---------
DEFAULT_INPUT_DIR="raw_vids"
DEFAULT_OUTPUT_DIR="output"

INPUT_DIR="${1:-}"
OUTPUT_DIR="${2:-}"

# --------- Functions ---------

log() {
    echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') - $*"
}

error_exit() {
    echo "[ERROR] $*" >&2
    exit 1
}

# --------- Argument Logging ---------

if [[ -n "$INPUT_DIR" ]]; then
    log "Using INPUT_DIR: $INPUT_DIR"
    if [[ ! -d "$INPUT_DIR" ]]; then
        error_exit "Input directory '$INPUT_DIR' does not exist!"
    fi
else
    log "No INPUT_DIR provided, using dataset/write_list.py defaults"
fi

if [[ -n "$OUTPUT_DIR" ]]; then
    log "Using OUTPUT_DIR: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
else
    log "No OUTPUT_DIR provided, using shubert_inference.py defaults"
fi

# --------- Conda Initialization ---------

if ! command -v conda >/dev/null 2>&1; then
    error_exit "Conda not found. Please install Anaconda or Miniconda."
fi

eval "$(conda shell.bash hook)"

# --------- Feature Extraction ---------

log "Creating conda environment: shubert_feature_extraction"
conda env create -f environment_feature_extraction.yml --name shubert_feature_extraction

log "Activating environment: shubert_feature_extraction"
conda activate shubert_feature_extraction

log "Running feature extraction pipeline..."

python download_models.py

log "Running write_list.py"
CMD=(python dataset/write_list.py)
if [[ -n "$INPUT_DIR" ]]; then
    CMD+=(--input-dir "$INPUT_DIR" --output-dir "raw_vids")
fi
log "Command: ${CMD[*]}"
"${CMD[@]}"

python dataset/clips_bbox.py
python dataset/kpe_mediapipe.py
python dataset/crop_hands.py
python dataset/crop_face.py
python features/body_features.py
python features/dinov2_features.py

log "Feature extraction completed."

# --------- SHuBERT Inference ---------

log "Creating conda environment: shubert_inference"
conda env create -f environment_shubert.yml --name shubert_inference

log "Activating environment: shubert_inference"
conda activate shubert_inference

log "Installing fairseq in editable mode"
cd fairseq
pip install -e .
cd -

log "Running SHuBERT inference"
CMD=(python features/shubert_inference.py)
if [[ -n "$OUTPUT_DIR" ]]; then
    CMD+=(--output-dir "$OUTPUT_DIR")
fi
log "Command: ${CMD[*]}"
"${CMD[@]}"

log "Pipeline completed successfully."
