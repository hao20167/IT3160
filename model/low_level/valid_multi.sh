#!/usr/bin/env bash
set -euo pipefail

NUM_GPUS=${NUM_GPUS:-1}
MAX_PARALLEL=${MAX_PARALLEL:-1}
LIMIT=${LIMIT:-50}

TEST_DF_PATH=${TEST_DF_PATH:-data/BTCUSDT/final/valid.feather}
PY=${PY:-python}
SCRIPT=${SCRIPT:-model/low_level/test_ddqn.py}

root_dirs=(
  "result_risk/BTCUSDT/beta_-10.0_risk_bond_0.1/seed_12345"
  "result_risk/BTCUSDT/beta_-90.0_risk_bond_0.1/seed_12345"
  "result_risk/BTCUSDT/beta_30.0_risk_bond_0.1/seed_12345"
  "result_risk/BTCUSDT/beta_100.0_risk_bond_0.1/seed_12345"
)

throttle() {
  while (( $(jobs -pr | wc -l) >= MAX_PARALLEL )); do
    sleep 0.5
  done
}

run_one_root() {
  local root_dir="$1"
  local beta_tag
  beta_tag="$(basename "$(dirname "$root_dir")")"
  local log_dir="log/pick/BTCUSDT/${beta_tag}"
  mkdir -p "$log_dir"

  # lấy list element + sort theo field 2 sau '_' (numeric)
  mapfile -t elements < <(
    for p in "$root_dir"/*; do
      [[ -e "$p" ]] || continue
      basename "$p"
    done | sort -t '_' -k2,2n
  )

  for action in 0 1 2 3 4; do
    local counter=0
    for element in "${elements[@]}"; do
      [[ "$element" == "log" ]] && continue

      local cuda_number=$((counter % NUM_GPUS))

      local epoch="${element#*_}"
      epoch="${epoch%%_*}"

      echo "dir=${beta_tag} action=${action} idx=${counter} epoch=${epoch} cuda=${cuda_number} element=${element}"

      local target_path="${root_dir}/${element}"
      local log_filename="${log_dir}/position_${action}_${element}.log"

      CUDA_VISIBLE_DEVICES="$cuda_number" "$PY" "$SCRIPT" \
        --test_path "$target_path" \
        --initial_action "$action" \
        --test_df_path "$TEST_DF_PATH" \
        >"$log_filename" 2>&1 &

      throttle
      ((counter++))
      ((counter >= LIMIT)) && break
    done
  done
}

main() {
  for dir in "${root_dirs[@]}"; do
    run_one_root "$dir"
  done
  wait
  echo "Done. Logs in log/pick/..."
}

main "$@"
