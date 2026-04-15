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
LOG_DIR="$ROOT_DIR/results/nightly_cross_ping_$TIMESTAMP"
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

printf 'Starting cross_ping batch across existing short/long dataset variants\n'
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
  RUN_CONFIG="$TMP_DIR/${VARIANT_NAME}_cross_ping_config.json"
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

dataset = str(paths.get("dataset", "unknown")).strip().lower()
dataset_paths = paths.get("dataset_paths", {})

cross_dataset = "unknown"
cross_data_dir = ""
cross_gold_labels = ""
cross_base_predictions = ""
if isinstance(dataset_paths, dict):
    normalized = {str(key).strip().lower(): value for key, value in dataset_paths.items()}
    if dataset in normalized:
        others = [name for name in sorted(normalized) if name != dataset]
        if len(others) == 1:
            cross_dataset = others[0]
            cross_entry = normalized.get(cross_dataset, {})
            if isinstance(cross_entry, dict):
                cross_data_dir = str(cross_entry.get("data_dir", ""))
                cross_gold_labels = str(cross_entry.get("gold_labels", ""))

prompt_overrides = loo.setdefault("prompt_kwargs_by_selection", {})
ping_prompt = dict(prompt_overrides.get("ping", {}))
cross_ping_prompt = dict(prompt_overrides.get("cross_ping", {}))
cross_ping_ra_prompt = dict(prompt_overrides.get("cross_ping_ra", {}))

merged_cross_ping_prompt = dict(ping_prompt)
merged_cross_ping_prompt.update(cross_ping_prompt)
prompt_overrides["cross_ping"] = merged_cross_ping_prompt

merged_cross_ping_ra_prompt = dict(merged_cross_ping_prompt)
merged_cross_ping_ra_prompt.update(cross_ping_ra_prompt)
prompt_overrides["cross_ping_ra"] = merged_cross_ping_ra_prompt

selection_overrides = loo.setdefault("selection_overrides", {})
ping_selection = dict(selection_overrides.get("ping", {}))
cross_ping_selection = dict(selection_overrides.get("cross_ping", {}))
cross_ping_ra_selection = dict(selection_overrides.get("cross_ping_ra", {}))

merged_cross_ping_selection = dict(ping_selection)
merged_cross_ping_selection.update(cross_ping_selection)
selection_overrides["cross_ping"] = merged_cross_ping_selection

merged_cross_ping_ra_selection = dict(merged_cross_ping_selection)
merged_cross_ping_ra_selection.update(cross_ping_ra_selection)
selection_overrides["cross_ping_ra"] = merged_cross_ping_ra_selection

loo["run_selections"] = ["cross_ping", "cross_ping_ra"]

output_root = str(paths.get("output_root", src.parent.parent))
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

if prompt_variant in {"short", "long"}:
    if dataset == "mimic":
        cross_dataset = "our"
        cross_data_dir = "data/india/internvl_anonymized_docs"
        cross_gold_labels = "data/india/internvl_gold_labels.pkl"
        cross_base_predictions = str(
            Path(f"results/mimic_streamlined_pipeline_small_models_indic_upd_{prompt_variant}")
            / zero_subdir
            / "all_models.pkl"
        )
    elif dataset == "our":
        cross_dataset = "mimic"
        cross_data_dir = "data/mimic_obs/annotated_merged"
        cross_gold_labels = "data/mimic_obs/annotated_merged_gold_upd.pkl"
        cross_base_predictions = str(
            Path(f"results/mimic_streamlined_pipeline_small_models_upd_{prompt_variant}")
            / zero_subdir
            / "all_models.pkl"
        )

if isinstance(dataset_paths, dict) and cross_dataset != "unknown":
    actual_key = None
    for raw_key in dataset_paths.keys():
        if str(raw_key).strip().lower() == cross_dataset:
            actual_key = raw_key
            break
    if actual_key is None:
        actual_key = cross_dataset
    cross_entry = dataset_paths.setdefault(actual_key, {})
    if isinstance(cross_entry, dict):
        if cross_data_dir:
            cross_entry["data_dir"] = cross_data_dir
        if cross_gold_labels:
            cross_entry["gold_labels"] = cross_gold_labels
        if cross_base_predictions:
            cross_entry["base_predictions_path"] = cross_base_predictions

dst.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")

print(output_root)
print(dataset)
print(prompt_variant)
print(cross_dataset)
print(cross_data_dir)
print(cross_gold_labels)
print(cross_base_predictions)
PY
  )

  OUTPUT_ROOT="${CFG_VALUES[0]:-}"
  DATASET_NAME="${CFG_VALUES[1]:-unknown}"
  PROMPT_VARIANT="${CFG_VALUES[2]:-unknown}"
  CROSS_DATASET_NAME="${CFG_VALUES[3]:-unknown}"
  CROSS_DATA_DIR="${CFG_VALUES[4]:-}"
  CROSS_GOLD_LABELS="${CFG_VALUES[5]:-}"
  CROSS_BASE_PREDICTIONS="${CFG_VALUES[6]:-}"
  LIVE_SNAPSHOT="$ROOT_DIR/$SNAPSHOT"
  LOG_FILE="$LOG_DIR/${VARIANT_NAME}.log"

  printf '[%s] Running variant=%s dataset=%s prompt=%s\n' \
    "$(date '+%Y-%m-%d %H:%M:%S')" "$VARIANT_NAME" "$DATASET_NAME" "$PROMPT_VARIANT"
  printf '[%s] Output root=%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$OUTPUT_ROOT"
  printf '[%s] Cross source dataset=%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$CROSS_DATASET_NAME"
  printf '[%s] Cross source data_dir=%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$CROSS_DATA_DIR"
  printf '[%s] Cross source gold_labels=%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$CROSS_GOLD_LABELS"
  printf '[%s] Cross source base_predictions=%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$CROSS_BASE_PREDICTIONS"

  if [[ $DRY_RUN -eq 0 ]]; then
    if [[ -n "$CROSS_DATA_DIR" && ! -d "$CROSS_DATA_DIR" ]]; then
      printf '[%s] MISSING cross source data_dir for %s: %s\n\n' \
        "$(date '+%Y-%m-%d %H:%M:%S')" "$VARIANT_NAME" "$CROSS_DATA_DIR"
      FAILED=$((FAILED + 1))
      continue
    fi
    if [[ -n "$CROSS_GOLD_LABELS" && ! -f "$CROSS_GOLD_LABELS" ]]; then
      printf '[%s] MISSING cross source gold labels for %s: %s\n\n' \
        "$(date '+%Y-%m-%d %H:%M:%S')" "$VARIANT_NAME" "$CROSS_GOLD_LABELS"
      FAILED=$((FAILED + 1))
      continue
    fi
    if [[ -n "$CROSS_BASE_PREDICTIONS" && ! -f "$CROSS_BASE_PREDICTIONS" ]]; then
      printf '[%s] MISSING cross source base predictions for %s: %s\n\n' \
        "$(date '+%Y-%m-%d %H:%M:%S')" "$VARIANT_NAME" "$CROSS_BASE_PREDICTIONS"
      FAILED=$((FAILED + 1))
      continue
    fi
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
for selection in (
    list(original_loo.get("run_selections", []))
    + list(current_loo.get("run_selections", []))
    + ["cross_ping", "cross_ping_ra"]
):
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

    ping_base = dict(merged.get("ping", {})) if isinstance(merged.get("ping"), dict) else {}
    cross_ping_cfg = dict(merged.get("cross_ping", {})) if isinstance(merged.get("cross_ping"), dict) else {}
    cross_ping_ra_cfg = dict(merged.get("cross_ping_ra", {})) if isinstance(merged.get("cross_ping_ra"), dict) else {}

    effective_cross_ping = dict(ping_base)
    effective_cross_ping.update(cross_ping_cfg)
    merged["cross_ping"] = effective_cross_ping

    effective_cross_ping_ra = dict(effective_cross_ping)
    effective_cross_ping_ra.update(cross_ping_ra_cfg)
    merged["cross_ping_ra"] = effective_cross_ping_ra

    current_loo[section_name] = merged

live_path.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
PY

      python - "$OUTPUT_ROOT" "$BACKUP_SNAPSHOT" "$LOG_FILE" "$DATASET_NAME" "$CROSS_DATASET_NAME" "$PROMPT_VARIANT" "$CROSS_BASE_PREDICTIONS" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

output_root = Path(sys.argv[1])
metadata = {
    "source_snapshot": sys.argv[2],
    "log_file": sys.argv[3],
    "dataset": sys.argv[4],
    "cross_dataset": sys.argv[5],
    "prompt_variant": sys.argv[6],
    "cross_base_predictions_path": sys.argv[7],
    "selections": ["cross_ping", "cross_ping_ra"],
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
}
(output_root / "cross_ping_batch_metadata.json").write_text(
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
