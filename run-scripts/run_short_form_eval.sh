#!/bin/bash

MODEL_NAME_OR_PATH=$1
WANDB_NAME=$2
DATASET_NAME=$3
DATASET_CONFIG_NAME=$4
DATASET_SPLIT_NAME=$5
TEXT_COLUMN_NAME=$6
LANGUAGE=$7

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_FILE="${PARENT_DIR}/logs/logs_eval_$(date +%Y-%m-%d_%H-%M-%S).txt"

mkdir -p "${PARENT_DIR}/logs"

export CUDA_VISIBLE_DEVICES=2

# Running the model evaluation with specified parameters
python "${PARENT_DIR}/distil-whisper/training/run_eval.py"  \
--model_name_or_path "$MODEL_NAME_OR_PATH" \
--dataset_name "$DATASET_NAME" \
--dataset_config_name "$DATASET_CONFIG_NAME" \
--dataset_split_name "$DATASET_SPLIT_NAME" \
--text_column_name "$TEXT_COLUMN_NAME" \
--batch_size 32 \
--dtype "float32" \
--generation_max_length 256 \
--language "$LANGUAGE" \
--attn_implementation "sdpa" \
--only_short_form \
--streaming True \
--wandb_name "$WANDB_NAME" >"$LOG_FILE" 2>&1