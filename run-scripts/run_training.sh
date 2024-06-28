#!/bin/bash

# Usage: ./run_training.sh [MODEL_NAME_OR_PATH] [TRAIN_DATASET_NAME] [TRAIN_SPLIT_NAME] [TEXT_COLUMN_NAME] [EVAL_DATASET_NAME] [EVAL_SPLIT_NAME] [EVAL_TEXT_COLUMN_NAME] [WARMUP_STEPS] [LEARNING_RATE] [TIMESTAMP_PROBABILITY] [CONDITION_ON_PREV_PROBABILITY] [LANGUAGE] [MAX_STEPS] [WER_THRESHOLD] [PER_DEVICE_TRAIN_BATCH_SIZE] [PER_DEVICE_EVAL_BATCH_SIZE] [DATALOADER_NUM_WORKERS] [WANDB_NAME]

MODEL_NAME_OR_PATH=$1
TRAIN_DATASET_NAME=$2
TRAIN_SPLIT_NAME=$3
TEXT_COLUMN_NAME=$4
EVAL_DATASET_NAME=$5
EVAL_SPLIT_NAME=$6
EVAL_TEXT_COLUMN_NAME=$7
WARMUP_STEPS=$8
LEARNING_RATE=$9
TIMESTAMP_PROBABILITY=${10}
CONDITION_ON_PREV_PROBABILITY=${11}
LANGUAGE=${12}
MAX_STEPS=${13}
WER_THRESHOLD=${14}
PER_DEVICE_TRAIN_BATCH_SIZE=${15}
PER_DEVICE_EVAL_BATCH_SIZE=${16}
DATALOADER_NUM_WORKERS=${17}
OUTPUT_DIR=${18}
WANDB_NAME=${19}

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
LOG_FILE="${SCRIPT_DIR}/logs/log_training_$(date +%Y-%m-%d_%H-%M-%S).txt"

accelerate launch --config_file "${SCRIPT_DIR}/../accelerate-configs/1gpu_config.yaml" \
"${SCRIPT_DIR}/../distil-whisper/training/run_distillation.py" \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --teacher_model_name_or_path "openai/whisper-large-v3" \
  --train_dataset_name "$TRAIN_DATASET_NAME" \
  --train_split_name "$TRAIN_SPLIT_NAME" \
  --text_column_name "$TEXT_COLUMN_NAME" \
  --eval_dataset_name "$EVAL_DATASET_NAME" \
  --eval_split_name "$EVAL_SPLIT_NAME" \
  --eval_text_column_name "$EVAL_TEXT_COLUMN_NAME" \
  --eval_steps 1000 \
  --save_steps 1000 \
  --warmup_steps "$WARMUP_STEPS" \
  --learning_rate "$LEARNING_RATE" \
  --lr_scheduler_type "linear" \
  --dtype "bfloat16" \
  --temperature 2.0 \
  --max_grad_norm 1.0 \
  --adam_beta1 0.9 \
  --adam_beta2 0.999 \
  --adam_epsilon 0.00000001 \
  --weight_decay 0.0 \
  --timestamp_probability "$TIMESTAMP_PROBABILITY" \
  --condition_on_prev_probability "$CONDITION_ON_PREV_PROBABILITY" \
  --language "$LANGUAGE" \
  --task "transcribe" \
  --logging_steps 25 \
  --save_total_limit 1 \
  --max_steps "$MAX_STEPS" \
  --wer_threshold "$WER_THRESHOLD" \
  --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
  --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE" \
  --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
  --ddp_timeout 7200 \
  --attn_implementation "sdpa" \
  --output_dir "$OUTPUT_DIR" \
  --do_train \
  --do_eval \
  --gradient_checkpointing \
  --overwrite_output_dir \
  --predict_with_generate True \
  --freeze_encoder \
  --freeze_embed_positions \
  --streaming True \
  --push_to_hub True \
  --wandb_name "$WANDB_NAME" \
  --wandb_dir /home/user/wandb \
  --torch_compile True >"$LOG_FILE" 2>&1