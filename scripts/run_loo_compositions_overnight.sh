#!/usr/bin/env bash
set -u -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG_PATH="configs/mimic_baseline_loo_config.json"
if [[ $# -gt 0 && "${1:0:1}" != "-" ]]; then
  CONFIG_PATH="$1"
  shift
fi

EXTRA_ARGS=("$@")
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT_DIR/results/nightly_loo_compositions_$TIMESTAMP"
mkdir -p "$LOG_DIR"

COMPOSITIONS=(hi3 lo3 mix3)
FAILED=0

printf 'Starting overnight LOO composition batch\n'
printf 'Repo: %s\n' "$ROOT_DIR"
printf 'Config: %s\n' "$CONFIG_PATH"
printf 'Logs: %s\n\n' "$LOG_DIR"

for COMP in "${COMPOSITIONS[@]}"; do
  LOG_FILE="$LOG_DIR/${COMP}.log"
  printf '[%s] Running composition=%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$COMP"
  printf '[%s] Command: python run_mimic_baseline_and_loo.py --config %s --stage loo --composition %s %s\n' \
    "$(date '+%Y-%m-%d %H:%M:%S')" "$CONFIG_PATH" "$COMP" "${EXTRA_ARGS[*]:-}"

  if python run_mimic_baseline_and_loo.py \
      --config "$CONFIG_PATH" \
      --stage loo \
      --composition "$COMP" \
      "${EXTRA_ARGS[@]}" 2>&1 | tee "$LOG_FILE"; then
    printf '[%s] Completed composition=%s (log: %s)\n\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$COMP" "$LOG_FILE"
  else
    printf '[%s] FAILED composition=%s (log: %s)\n\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$COMP" "$LOG_FILE"
    FAILED=$((FAILED + 1))
  fi

done

if [[ $FAILED -gt 0 ]]; then
  printf '[%s] Batch finished with %d failed run(s).\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$FAILED"
  exit 1
fi

printf '[%s] Batch finished successfully.\n' "$(date '+%Y-%m-%d %H:%M:%S')"
