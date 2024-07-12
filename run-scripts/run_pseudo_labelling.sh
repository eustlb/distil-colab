#!/bin/bash

# Usage: ./script_name.sh <model_name_or_path> <dataset_name> <dataset_config_name> <dataset_split_name> <audio_column_name> <text_column_name> <language> <id_column_name>

MODEL_NAME_OR_PATH=$1
DATASET_NAME=$2
DATASET_CONFIG_NAME=$3
DATASET_SPLIT_NAME=$4
AUDIO_COLUMN_NAME=$5
TEXT_COLUMN_NAME=$6
LANGUAGE=$7
ID_COLUMN_NAME=$8

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${PARENT_DIR}/tmp/${DATASET_NAME}_${LANGUAGE}_pseudo_labelled"
LOG_FILE="${PARENT_DIR}/logs/logs_pseudo_labelling_$(date +%Y-%m-%d_%H-%M-%S).txt"

mkdir -p "${PARENT_DIR}/tmp"
mkdir -p "${PARENT_DIR}/logs"
export HF_DATASETS_CACHE="/data/.cache"

accelerate launch --config_file "${PARENT_DIR}/accelerate-configs/1gpu_config.yaml" \
"${PARENT_DIR}/distil-whisper/training/run_pseudo_labelling.py" \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --dataset_name "$DATASET_NAME" \
  --dataset_config_name "$DATASET_CONFIG_NAME" \
  --dataset_split_name "$DATASET_SPLIT_NAME" \
  --audio_column_name "$AUDIO_COLUMN_NAME" \
  --text_column_name "$TEXT_COLUMN_NAME" \
  --id_column_name "$ID_COLUMN_NAME" \
  --output_dir "$OUTPUT_DIR" \
  --wandb_project "distil-whisper-labelling" \
  --per_device_eval_batch_size 32 \
  --dtype "bfloat16" \
  --attn_implementation "sdpa" \
  --logging_steps 500 \
  --max_label_length 256 \
  --concatenate_audio \
  --preprocessing_batch_size 500 \
  --preprocessing_num_workers 48 \
  --dataloader_num_workers 1 \
  --report_to "wandb" \
  --language "$LANGUAGE" \
  --task "transcribe" \
  --return_timestamps \
  --streaming True \
  --generation_num_beams 1 \
  --push_to_hub False \
  --ddp_timeout 7200 >"$LOG_FILE" 2>&1