#!/bin/bash
# =============================================================================
# Train the PUMA Redundancy Detector (RD):
#   full fine-tune Qwen3-Embedding-0.6B with an InfoNCE contrastive objective.
#
# Requires ms-swift (https://github.com/modelscope/ms-swift).
#
# Data: a directory with train.jsonl + dev.jsonl in the contrastive format
#   {messages, positive_messages, negative_messages}.
#   Full data: https://huggingface.co/datasets/ZhishanQ/puma-rd-training-data
#   A 30-row format sample is in train_rd/sample_train.jsonl.
#
# Usage:
#   bash train_rd/train_rd.sh <data_dir> <output_dir> [nproc_per_node]
#
# Example:
#   bash train_rd/train_rd.sh data/rd/v9_corrected_filt output/rd_full 3
# =============================================================================
set -euo pipefail

DATA_DIR="${1:?Usage: train_rd.sh <data_dir> <output_dir> [nproc_per_node]}"
OUTPUT_DIR="${2:?Usage: train_rd.sh <data_dir> <output_dir> [nproc_per_node]}"
NPROC="${3:-1}"

# ---- InfoNCE configuration ----
export INFONCE_USE_BATCH=true          # in-batch negatives
export INFONCE_MASK_FAKE_NEGATIVE=true
export INFONCE_TEMPERATURE=0.05
export INFONCE_HARD_NEGATIVES=8        # hard negatives per anchor used during training
export NPROC_PER_NODE="$NPROC"

swift sft \
  --model Qwen/Qwen3-Embedding-0.6B \
  --task_type embedding \
  --model_type qwen3_emb \
  --train_type full \
  --dataset "${DATA_DIR}/train.jsonl" \
  --val_dataset "${DATA_DIR}/dev.jsonl" \
  --loss_type infonce \
  --max_length 2048 \
  --truncation_strategy delete \
  --output_dir "${OUTPUT_DIR}" \
  --num_train_epochs 5 \
  --per_device_train_batch_size 12 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --learning_rate 5e-5 \
  --warmup_ratio 0.05 \
  --eval_strategy steps \
  --eval_steps 300 \
  --save_steps 300 \
  --save_total_limit 5 \
  --dataloader_drop_last true \
  --dataloader_num_workers 8 \
  --logging_steps 10 \
  --load_best_model_at_end true \
  --metric_for_best_model eval_loss \
  --greater_is_better false \
  --bf16 true \
  --attn_impl flash_attention_2 \
  --gradient_checkpointing true

echo "Done. RD checkpoint(s) saved under: ${OUTPUT_DIR}"
