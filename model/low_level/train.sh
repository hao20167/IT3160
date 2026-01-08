#!/bin/bash
set -e

DATASET="BTCUSDT"
TRAIN_PATH="data/${DATASET}/final/train"
LOG_DIR="log/train/${DATASET}/low_level"

mkdir -p "${LOG_DIR}"

# Nếu Colab chỉ có 1 GPU thì CUDA_VISIBLE_DEVICES luôn = 0
GPU_ID=0

BETAS=(100 -10 -90 30)

for BETA in "${BETAS[@]}"; do
  LOG_FILE="${LOG_DIR}/beta_${BETA}.log"
  echo "=============================="
  echo "Starting beta=${BETA} on GPU ${GPU_ID}"
  echo "Log: ${LOG_FILE}"
  echo "=============================="

  CUDA_VISIBLE_DEVICES=${GPU_ID} python model/low_level/ddqn_pes_risk_aware.py \
    --beta "${BETA}" \
    --train_data_path "${TRAIN_PATH}" \
    --dataset_name "${DATASET}" \
    > "${LOG_FILE}" 2>&1

  echo "Finished beta=${BETA}"
done

echo "All jobs completed."