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
LOG_DIR="$ROOT_DIR/results/nightly_ping_ablations_$TIMESTAMP"
mkdir -p "$LOG_DIR"

mapfile -t CFG_VALUES < <(
  python - "$CONFIG_PATH" <<'PY'
import json
import sys
from pathlib import Path

cfg_path = Path(sys.argv[1])
if not cfg_path.exists():
    raise SystemExit(f"Config not found: {cfg_path}")
cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
paths = cfg.get("paths", {})
base_output_root = paths.get("output_root", "results/mimic_pipeline")
zero_subdir = cfg.get("zero_shot", {}).get("output_subdir", "zero_shot")
shared_base_predictions = paths.get("base_predictions_path")
if not shared_base_predictions:
    shared_base_predictions = str(Path(base_output_root) / zero_subdir / "all_models.pkl")
print(base_output_root)
print(shared_base_predictions)
PY
)
BASE_OUTPUT_ROOT="${CFG_VALUES[0]:-}"
SHARED_BASE_PREDICTIONS="${CFG_VALUES[1]:-}"
if [[ -z "$BASE_OUTPUT_ROOT" ]]; then
  BASE_OUTPUT_ROOT="results/mimic_pipeline"
fi
if [[ -z "$SHARED_BASE_PREDICTIONS" ]]; then
  SHARED_BASE_PREDICTIONS="${BASE_OUTPUT_ROOT}/zero_shot/all_models.pkl"
fi

DRY_RUN=0
for ARG in "${EXTRA_ARGS[@]}"; do
  if [[ "$ARG" == "--dry-run" ]]; then
    DRY_RUN=1
    break
  fi
done

if [[ $DRY_RUN -eq 0 && ! -f "$SHARED_BASE_PREDICTIONS" ]]; then
  printf 'ERROR: base predictions not found: %s\n' "$SHARED_BASE_PREDICTIONS"
  printf 'Run zero-shot first or pass --base-predictions-path via EXTRA_ARGS.\n'
  exit 1
fi




ABLATIONS=(
  a6_granule_prior_round_robin_no_pong
  a7_ping_only_reveal_no_compensation
  a8_reveal_yhat_round_robin_no_pong
  a3_uncertainty_only_no_pong
  a4_diversity_only_no_pong
  a5_harmonic_no_pong

)

FAILED=0

printf 'Starting ping ablation batch\n'
printf 'Repo: %s\n' "$ROOT_DIR"
printf 'Config: %s\n' "$CONFIG_PATH"
printf 'Base output root: %s\n' "$BASE_OUTPUT_ROOT"
printf 'Shared base predictions: %s\n' "$SHARED_BASE_PREDICTIONS"
printf 'Logs: %s\n\n' "$LOG_DIR"

for ABLATION in "${ABLATIONS[@]}"; do
  OUT_ROOT="${BASE_OUTPUT_ROOT}_${ABLATION}"
  LOG_FILE="$LOG_DIR/${ABLATION}.log"

  printf '[%s] Running ablation=%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$ABLATION"
  printf '[%s] Output root=%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$OUT_ROOT"
  printf '[%s] Command: python run_mimic_baseline_and_loo.py --config %s --stage loo --ping-ablation %s --output-root %s --base-predictions-path %s %s\n' \
    "$(date '+%Y-%m-%d %H:%M:%S')" "$CONFIG_PATH" "$ABLATION" "$OUT_ROOT" "$SHARED_BASE_PREDICTIONS" "${EXTRA_ARGS[*]:-}"

  if python run_mimic_baseline_and_loo.py \
      --config "$CONFIG_PATH" \
      --stage loo \
      --ping-ablation "$ABLATION" \
      --output-root "$OUT_ROOT" \
      --base-predictions-path "$SHARED_BASE_PREDICTIONS" \
      "${EXTRA_ARGS[@]}" 2>&1 | tee "$LOG_FILE"; then
    python - "$OUT_ROOT" "$ABLATION" "$CONFIG_PATH" "$LOG_FILE" "$SHARED_BASE_PREDICTIONS" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

out_root = Path(sys.argv[1])
meta = {
    "ablation": sys.argv[2],
    "source_config": sys.argv[3],
    "log_file": sys.argv[4],
    "base_predictions_path": sys.argv[5],
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "config_snapshot": str(out_root / "config_snapshot.json"),
}
out_root.mkdir(parents=True, exist_ok=True)
with (out_root / "ablation_metadata.json").open("w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)
PY
    printf '[%s] Completed ablation=%s (log: %s)\n\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$ABLATION" "$LOG_FILE"
  else
    printf '[%s] FAILED ablation=%s (log: %s)\n\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$ABLATION" "$LOG_FILE"
    FAILED=$((FAILED + 1))
  fi
done

if [[ $FAILED -gt 0 ]]; then
  printf '[%s] Batch finished with %d failed run(s).\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$FAILED"
  exit 1
fi

printf '[%s] Batch finished successfully.\n' "$(date '+%Y-%m-%d %H:%M:%S')"
