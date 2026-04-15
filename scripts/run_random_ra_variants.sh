#!/usr/bin/env bash
set -u -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TARGET_SNAPSHOTS=(
  "results/mimic_streamlined_pipeline_small_models_upd_short/config_snapshot.json"
  "results/mimic_streamlined_pipeline_small_models_upd_long/config_snapshot.json"
  "results/mimic_streamlined_pipeline_small_models_indic_upd_short/config_snapshot.json"
  "results/mimic_streamlined_pipeline_small_models_indic_upd_long/config_snapshot.json"
)

EXTRA_ARGS=("$@")
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT_DIR/results/nightly_random_ra_$TIMESTAMP"
TMP_DIR="$(mktemp -d)"
mkdir -p "$LOG_DIR"
trap 'rm -rf "$TMP_DIR"' EXIT

for ARG in "${EXTRA_ARGS[@]}"; do
  case "$ARG" in
    --config|--config=*|--stage|--stage=*|--output-root|--output-root=*|--base-predictions-path|--base-predictions-path=*)
      printf 'ERROR: %s is managed by this script and may not be passed through.\n' "$ARG"
      exit 2
      ;;
  esac
done

DRY_RUN=0
for ARG in "${EXTRA_ARGS[@]}"; do
  if [[ "$ARG" == "--dry-run" ]]; then
    DRY_RUN=1
    break
  fi
done

FAILED=0

printf 'Starting random_ra batch across existing short/long dataset variants\n'
printf 'Repo: %s\n' "$ROOT_DIR"
printf 'Logs: %s\n' "$LOG_DIR"
printf 'Targets:\n'
for SNAPSHOT in "${TARGET_SNAPSHOTS[@]}"; do
  printf '  - %s\n' "$SNAPSHOT"
done
printf '\n'

for SNAPSHOT in "${TARGET_SNAPSHOTS[@]}"; do
  if [[ ! -f "$SNAPSHOT" ]]; then
    printf '[%s] MISSING snapshot: %s\n\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$SNAPSHOT"
    FAILED=$((FAILED + 1))
    continue
  fi

  VARIANT_NAME="$(basename "$(dirname "$SNAPSHOT")")"
  RUN_CONFIG="$TMP_DIR/${VARIANT_NAME}_random_ra_config.json"
  BACKUP_SNAPSHOT="$TMP_DIR/${VARIANT_NAME}_config_snapshot.backup.json"
  cp "$SNAPSHOT" "$BACKUP_SNAPSHOT"

  mapfile -t CFG_VALUES < <(
    python - "$BACKUP_SNAPSHOT" "$RUN_CONFIG" <<'PY'
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
cfg = json.loads(src.read_text(encoding="utf-8"))

paths = cfg.setdefault("paths", {})
loo = cfg.setdefault("loo", {})

prompt_overrides = loo.setdefault("prompt_kwargs_by_selection", {})
random_prompt = dict(prompt_overrides.get("random", {}))
random_ra_prompt = dict(prompt_overrides.get("random_ra", {}))
merged_prompt = dict(random_prompt)
merged_prompt.update(random_ra_prompt)
prompt_overrides["random_ra"] = merged_prompt

selection_overrides = loo.setdefault("selection_overrides", {})
random_selection = dict(selection_overrides.get("random", {}))
random_ra_selection = dict(selection_overrides.get("random_ra", {}))
merged_selection = dict(random_selection)
merged_selection.update(random_ra_selection)
selection_overrides["random_ra"] = merged_selection

loo["run_selections"] = ["random_ra"]

output_root = str(paths.get("output_root", src.parent.parent))
dataset = str(paths.get("dataset", "unknown"))
prompt_variant = str(
    loo.get("prompt_option")
    or cfg.get("zero_shot", {}).get("prompt_option")
    or "unknown"
)
if prompt_variant == "unknown":
    output_root_lower = output_root.lower()
    if "_short" in output_root_lower:
        prompt_variant = "short"
    elif "_long" in output_root_lower:
        prompt_variant = "long"

zero_subdir = str(cfg.get("zero_shot", {}).get("output_subdir", "zero_shot"))
base_predictions = str(Path(output_root) / zero_subdir / "all_models.pkl")

dst.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")

print(output_root)
print(dataset)
print(prompt_variant)
print(base_predictions)
PY
  )

  OUTPUT_ROOT="${CFG_VALUES[0]:-}"
  DATASET_NAME="${CFG_VALUES[1]:-unknown}"
  PROMPT_VARIANT="${CFG_VALUES[2]:-unknown}"
  BASE_PREDICTIONS="${CFG_VALUES[3]:-}"
  LIVE_SNAPSHOT="$ROOT_DIR/$SNAPSHOT"
  LOG_FILE="$LOG_DIR/${VARIANT_NAME}.log"

  printf '[%s] Running variant=%s dataset=%s prompt=%s\n' \
    "$(date '+%Y-%m-%d %H:%M:%S')" "$VARIANT_NAME" "$DATASET_NAME" "$PROMPT_VARIANT"
  printf '[%s] Output root=%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$OUTPUT_ROOT"
  printf '[%s] Base predictions=%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$BASE_PREDICTIONS"

  if [[ $DRY_RUN -eq 0 && ! -f "$BASE_PREDICTIONS" ]]; then
    printf '[%s] MISSING base predictions for %s: %s\n\n' \
      "$(date '+%Y-%m-%d %H:%M:%S')" "$VARIANT_NAME" "$BASE_PREDICTIONS"
    FAILED=$((FAILED + 1))
    continue
  fi

  printf '[%s] Command: python run_mimic_baseline_and_loo.py --config %s --stage loo %s\n' \
    "$(date '+%Y-%m-%d %H:%M:%S')" "$RUN_CONFIG" "${EXTRA_ARGS[*]:-}"

  if python run_mimic_baseline_and_loo.py \
      --config "$RUN_CONFIG" \
      --stage loo \
      "${EXTRA_ARGS[@]}" 2>&1 | tee "$LOG_FILE"; then
    if [[ $DRY_RUN -eq 0 ]]; then
      python - "$BACKUP_SNAPSHOT" "$LIVE_SNAPSHOT" <<'PY'
import json
import sys
from pathlib import Path

backup_path = Path(sys.argv[1])
live_path = Path(sys.argv[2])

original = json.loads(backup_path.read_text(encoding="utf-8"))
current = json.loads(live_path.read_text(encoding="utf-8")) if live_path.exists() else {}

original_loo = original.get("loo", {})
current_loo = current.setdefault("loo", {})

selection_order = []
for selection in list(original_loo.get("run_selections", [])) + list(current_loo.get("run_selections", [])) + ["random_ra"]:
    if selection not in selection_order:
        selection_order.append(selection)
current_loo["run_selections"] = selection_order

for section_name in ("prompt_kwargs_by_selection", "selection_overrides"):
    merged = {}
    original_section = original_loo.get(section_name, {})
    current_section = current_loo.get(section_name, {})
    if isinstance(original_section, dict):
        merged.update(original_section)
    if isinstance(current_section, dict):
        merged.update(current_section)
    if "random_ra" not in merged:
        base_random = merged.get("random", {})
        merged["random_ra"] = dict(base_random) if isinstance(base_random, dict) else {}
    current_loo[section_name] = merged

live_path.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
PY

      python - "$OUTPUT_ROOT" "$BACKUP_SNAPSHOT" "$LOG_FILE" "$DATASET_NAME" "$PROMPT_VARIANT" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

output_root = Path(sys.argv[1])
metadata = {
    "source_snapshot": sys.argv[2],
    "log_file": sys.argv[3],
    "dataset": sys.argv[4],
    "prompt_variant": sys.argv[5],
    "selection": "random_ra",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
}
(output_root / "random_ra_batch_metadata.json").write_text(
    json.dumps(metadata, indent=2) + "\n",
    encoding="utf-8",
)
PY
    fi

    printf '[%s] Completed variant=%s (log: %s)\n\n' \
      "$(date '+%Y-%m-%d %H:%M:%S')" "$VARIANT_NAME" "$LOG_FILE"
  else
    if [[ $DRY_RUN -eq 0 ]]; then
      cp "$BACKUP_SNAPSHOT" "$LIVE_SNAPSHOT"
    fi
    printf '[%s] FAILED variant=%s (log: %s)\n\n' \
      "$(date '+%Y-%m-%d %H:%M:%S')" "$VARIANT_NAME" "$LOG_FILE"
    FAILED=$((FAILED + 1))
  fi
done

if [[ $FAILED -gt 0 ]]; then
  printf '[%s] Batch finished with %d failed variant(s).\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$FAILED"
  exit 1
fi

printf '[%s] Batch finished successfully.\n' "$(date '+%Y-%m-%d %H:%M:%S')"
