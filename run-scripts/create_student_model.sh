#!/bin/bash

TEACHER_CHECKPOINT=$1
SAVE_DIR=$2

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
 
# Execute the Python script with provided arguments
python "${PARENT_DIR}/distil-whisper/training/create_student_model.py" \
  --teacher_checkpoint "$TEACHER_CHECKPOINT" \
  --encoder_layers 32 \
  --decoder_layers 2 \
  --save_dir "$SAVE_DIR"