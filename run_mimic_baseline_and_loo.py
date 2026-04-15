#!/usr/bin/env python3
"""Config-driven runner for MIMIC zero-shot + LOO ICL experiments.

This script streamlines the workflows currently executed in:
- baseline_experiments_v3_mimic_merged.ipynb
- baseline_experiments_loo_v3_mimic.ipynb

Usage
-----
python run_mimic_baseline_and_loo.py --config configs/mimic_baseline_loo_config.json
python run_mimic_baseline_and_loo.py --config configs/mimic_baseline_loo_config.json --stage zero-shot
python run_mimic_baseline_and_loo.py --config configs/mimic_baseline_loo_config.json --stage loo
python run_mimic_baseline_and_loo.py --config configs/mimic_baseline_loo_config.json --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from itertools import product
from pathlib import Path
from datetime import datetime, timezone
import traceback
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

try:
    import torch
except Exception:  # pragma: no cover - defensive fallback
    torch = None

from src.config_utils import resolve_dataset_paths as _resolve_dataset_paths_cfg
from src.data_loader import load_documents
from src.evaluation_utils import evaluate_by_model
from src.label_utils import canonicalize_label
from src.loo_selector import DEFAULT_QUERY_TEMPLATE, LeaveOneOutPromptBuilder, prepare_loo_resources
from src.model_runner import PROMPT_TEMPLATE as DEFAULT_ZERO_SHOT_PROMPT
from src.model_runner import run_zero_shot_classification

PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_ICL_INSTRUCTION = (
    "You are a medical assistant specialized in Obstetric and Gynecologic Ultrasound.\n"
    "\n"
    "Task: Classify the document-level ACTIVE pregnancy status at the time of this report, using ONLY what is stated or strongly implied in the report (no outside chart review).\n"
    "\n"
    "Choose exactly ONE label from:\n"
    "- early active pregnancy\n"
    "- late active pregnancy\n"
    "- no active pregnancy\n"
    "\n"
    "Key definition (ACTIVE pregnancy):\n"
    "\"Active\" means pregnancy tissue/trophoblastic tissue is currently present or reasonably suspected to be present such that pregnancy-specific evaluation/management is ongoing, regardless of viability. This includes:\n"
    "- confirmed/suspected intrauterine pregnancy (IUP), including nonviable IUP if management is not documented as complete,\n"
    "- confirmed/suspected ectopic pregnancy,\n"
    "- pregnancy of unknown location (PUL) when ongoing pregnancy/ectopic is still being worked up,\n"
    "- vascularized retained products of conception (RPOC) with suspected persistent trophoblastic activity.\n"
    "Do NOT switch to \"no active pregnancy\" at the moment nonviability is diagnosed; switch only when the report documents resolution/completion (e.g., confirmed passage, completed procedure, or follow-up showing no persistent active pregnancy).\n"
    "\n"
    "Decision rules (apply in order):\n"
    "1) Decide ACTIVE vs NO ACTIVE first:\n"
    "   - If the report raises ongoing pregnancy/ectopic/PUL as an active diagnostic possibility (even \"cannot exclude\", \"correlate with beta-hCG\", serial follow-up), classify as ACTIVE (usually early active pregnancy).\n"
    "   - If the report describes resolved/post-management state with no ongoing pregnancy concern (postpartum; completed miscarriage with passage or completed treatment; post-treatment ectopic follow-up without persistent concern; non-obstetric gyn imaging), classify as NO ACTIVE.\n"
    "\n"
    "2) If ACTIVE, split early vs late using best GA evidence:\n"
    "   - early active pregnancy: GA < 14 weeks\n"
    "   - late active pregnancy: GA >= 14 weeks\n"
    "   GA may come from explicit GA, EDD inference, LMP estimate, fetal biometry, or clear exam context.\n"
    "\n"
    "3) If ACTIVE but GA cannot be determined:\n"
    "   - Default to early active pregnancy unless the report clearly indicates second-trimester/later pregnancy (e.g., anatomy survey, amniocentesis, established pregnancy procedures).\n"
    "\n"
    "RPOC rule:\n"
    "- Vascularized RPOC with suspected persistent trophoblastic activity / ongoing pregnancy-related management concern -> ACTIVE (typically early).\n"
    "- Non-vascularized RPOC with no gestational sac in post-miscarriage/post-procedure follow-up -> NO ACTIVE.\n"
    "\n"
    "Only output the label (exactly one of the three). Do not output any other text. "
    "I will give you some examples first and then it will be your turn. Examples:\n\n"
)

DEFAULT_PING_COMPOSITION = "all"
DEFAULT_PING_COMMITTEE_SIZE = 3
PING_SELECTIONS = {"ping", "ping_ra", "cross_ping", "cross_ping_ra"}
CROSS_PING_SELECTIONS = {"cross_ping", "cross_ping_ra"}
RANDOM_SELECTIONS = {"random", "random_ra"}
PING_ABLATION_PRESETS: Dict[str, Dict[str, Any]] = {
    "default": {
        # Default ping now uses reveal + y-hat round-robin probabilistic sampling
        # with mismatch-driven probability updates, and no compensation turns.
        "score": "harmonic",
        "use_random_pong": True,
        "use_pred_round_robin": True,
        "reveal_one_by_one": True,
        "reveal_no_pong_compensation": True,
        "reveal_ping_pred_round_robin_prob_update": True,
    },
    "a3_uncertainty_only_no_pong": {
        "score": "uncertainty",
        "use_random_pong": False,
    },
    "a4_diversity_only_no_pong": {
        "score": "diversity",
        "use_random_pong": False,
    },
    "a5_harmonic_no_pong": {
        "score": "harmonic",
        "use_random_pong": False,
    },
    "a6_granule_prior_round_robin_no_pong": {
        "score": "uncertainty",
        "use_random_pong": False,
        "granule_prior_round_robin": True,
        "granule_prior_cap": 0.95,
    },
    "a7_ping_only_reveal_no_compensation": {
        # Ping-only reveal: no compensation turns. Continue pinging from unlabeled pool
        # until per-class quota is reached, discarding over-quota reveals.
        "score": "harmonic",
        "use_random_pong": False,
        "use_pred_round_robin": False,
        "global_pool_round_local_diversity_after_first": True,
        "reveal_one_by_one": True,
        "reveal_no_pong_compensation": True,
    },
    "a8_reveal_yhat_round_robin_no_pong": {
        # Reveal + y-hat round-robin (no compensation): cycle predicted-label buckets
        # and take the next available sample in each bucket (no U/D scoring in-bucket).
        "score": "harmonic",
        "use_random_pong": False,
        "use_pred_round_robin": True,
        "reveal_one_by_one": True,
        "reveal_no_pong_compensation": True,
        "reveal_ping_pred_round_robin": True,
    },
    "a9_reveal_yhat_round_robin_prob_update_no_pong": {
        # Reveal + y-hat round-robin probabilistic sampling from counts; on mismatch,
        # update nearby docs by moving probability mass from intended class to revealed class.
        "score": "harmonic",
        "use_random_pong": True,
        "use_pred_round_robin": True,
        "reveal_one_by_one": True,
        "reveal_no_pong_compensation": True,
        "reveal_ping_pred_round_robin_prob_update": True,
    },
}
PING_ABLATION_ALIASES: Dict[str, str] = {
    "none": "default",
    "a3": "a3_uncertainty_only_no_pong",
    "uncertainty_only_no_pong": "a3_uncertainty_only_no_pong",
    "no_pong_uncertainty": "a3_uncertainty_only_no_pong",
    "a4": "a4_diversity_only_no_pong",
    "diversity_only_no_pong": "a4_diversity_only_no_pong",
    "no_pong_diversity": "a4_diversity_only_no_pong",
    "a5": "a5_harmonic_no_pong",
    "harmonic_no_pong": "a5_harmonic_no_pong",
    "combined_no_pong": "a5_harmonic_no_pong",
    "no_pong_combined": "a5_harmonic_no_pong",
    "a6": "a6_granule_prior_round_robin_no_pong",
    "granule_prior_round_robin_no_pong": "a6_granule_prior_round_robin_no_pong",
    "granule_prior_no_pong": "a6_granule_prior_round_robin_no_pong",
    "round_robin_granule_prior_no_pong": "a6_granule_prior_round_robin_no_pong",
    "no_pong_granule_prior": "a6_granule_prior_round_robin_no_pong",
    "a7": "a7_ping_only_reveal_no_compensation",
    "ping_only_reveal_no_compensation": "a7_ping_only_reveal_no_compensation",
    "reveal_no_compensation": "a7_ping_only_reveal_no_compensation",
    "no_compensation_ping_reveal": "a7_ping_only_reveal_no_compensation",
    "a8": "a8_reveal_yhat_round_robin_no_pong",
    "reveal_yhat_round_robin_no_pong": "a8_reveal_yhat_round_robin_no_pong",
    "yhat_round_robin_no_pong": "a8_reveal_yhat_round_robin_no_pong",
    "reveal_one_by_one_yhat_round_robin": "a8_reveal_yhat_round_robin_no_pong",
    "a9": "a9_reveal_yhat_round_robin_prob_update_no_pong",
    "reveal_yhat_round_robin_prob_update_no_pong": "a9_reveal_yhat_round_robin_prob_update_no_pong",
    "yhat_round_robin_prob_update_no_pong": "a9_reveal_yhat_round_robin_prob_update_no_pong",
    "reveal_one_by_one_yhat_round_robin_prob_update": "a9_reveal_yhat_round_robin_prob_update_no_pong",
}


def _normalize_ping_ablation(raw_value: str) -> str:
    normalized = re.sub(r"[\s\-_]+", "_", raw_value.strip().lower())
    normalized = normalized.strip("_")
    return PING_ABLATION_ALIASES.get(normalized, normalized)


def _apply_ping_ablation(loo_cfg: Dict[str, Any], raw_value: Optional[str]) -> Optional[str]:
    if raw_value is None:
        return None
    canonical = _normalize_ping_ablation(str(raw_value))
    if canonical not in PING_ABLATION_PRESETS:
        valid = ", ".join(sorted(PING_ABLATION_PRESETS.keys()))
        raise ValueError(f"Unsupported ping ablation '{raw_value}'. Supported values: {valid}.")

    overrides = dict(PING_ABLATION_PRESETS[canonical])
    selection_overrides = dict(loo_cfg.get("selection_overrides", {}))
    ping_overrides = dict(selection_overrides.get("ping", {}))
    ping_overrides.update(overrides)
    selection_overrides["ping"] = ping_overrides
    loo_cfg["selection_overrides"] = selection_overrides
    loo_cfg["ping_ablation"] = canonical
    return canonical


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_prompt_option(raw_value: Any) -> str:
    return str(raw_value).strip().lower()


def _resolve_prompt_option_value(
    root_cfg: Dict[str, Any],
    section_cfg: Dict[str, Any],
    *,
    options_key: str,
    explicit_value: Optional[str],
    default_value: str,
    section_name: str,
) -> str:
    raw_value = section_cfg.get("prompt_option", root_cfg.get("prompt_option"))
    if raw_value is not None:
        options = section_cfg.get(options_key)
        if not isinstance(options, dict) or not options:
            raise ValueError(
                f"{section_name}.{options_key} must be defined when prompt_option is set."
            )
        option = _normalize_prompt_option(raw_value)
        if option not in options:
            valid = ", ".join(sorted(str(k) for k in options.keys()))
            raise ValueError(
                f"Unsupported prompt_option '{raw_value}' for {section_name}. "
                f"Supported values: {valid}."
            )
        value = options[option]
        if not isinstance(value, str):
            raise ValueError(
                f"{section_name}.{options_key}.{option} must be a string prompt."
            )
        return value
    if explicit_value is not None:
        return str(explicit_value)
    return default_value


def _resolve_zero_shot_prompt_template(
    root_cfg: Dict[str, Any],
    zero_cfg: Dict[str, Any],
) -> str:
    return _resolve_prompt_option_value(
        root_cfg,
        zero_cfg,
        options_key="prompt_options",
        explicit_value=zero_cfg.get("prompt_template"),
        default_value=DEFAULT_ZERO_SHOT_PROMPT,
        section_name="zero_shot",
    )


def _resolve_icl_instruction(
    root_cfg: Dict[str, Any],
    loo_cfg: Dict[str, Any],
) -> str:
    prompt_kwargs = loo_cfg.get("prompt_kwargs", {})
    return _resolve_prompt_option_value(
        root_cfg,
        loo_cfg,
        options_key="prompt_options",
        explicit_value=prompt_kwargs.get("instruction"),
        default_value=DEFAULT_ICL_INSTRUCTION,
        section_name="loo",
    )


def _resolve_dataset_paths(
    paths_cfg: Dict[str, Any],
    *,
    base_dir: Optional[Path] = None,
) -> Tuple[Path, Optional[Path], Dict[str, Any]]:
    return _resolve_dataset_paths_cfg(paths_cfg, base_dir=base_dir)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _is_ping_selection(selection: str) -> bool:
    return str(selection).strip().lower() in PING_SELECTIONS


def _is_random_selection(selection: str) -> bool:
    return str(selection).strip().lower() in RANDOM_SELECTIONS


def _is_cross_ping_selection(selection: str) -> bool:
    return str(selection).strip().lower() in CROSS_PING_SELECTIONS


def _dataset_paths_lookup(paths_cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    dataset_paths = paths_cfg.get("dataset_paths")
    if not isinstance(dataset_paths, dict) or not dataset_paths:
        raise ValueError(
            "Cross-dataset ping requires Config.paths.dataset_paths with entries for both datasets."
        )

    lookup: Dict[str, Dict[str, Any]] = {}
    for raw_name, raw_entry in dataset_paths.items():
        dataset_name = str(raw_name).strip().lower()
        if not isinstance(raw_entry, dict):
            raise ValueError(
                f"Config.paths.dataset_paths['{raw_name}'] must be an object."
            )
        lookup[dataset_name] = dict(raw_entry)
    return lookup


def _resolve_cross_dataset_name(paths_cfg: Dict[str, Any]) -> str:
    current_dataset = str(paths_cfg.get("dataset", "")).strip().lower()
    dataset_lookup = _dataset_paths_lookup(paths_cfg)
    if current_dataset not in dataset_lookup:
        available = ", ".join(sorted(dataset_lookup))
        raise ValueError(
            f"Active dataset '{current_dataset}' is missing from Config.paths.dataset_paths. "
            f"Available: {available}"
        )

    other_datasets = [name for name in sorted(dataset_lookup) if name != current_dataset]
    if len(other_datasets) != 1:
        raise ValueError(
            "Cross-dataset ping requires exactly one opposite dataset in "
            f"Config.paths.dataset_paths; found {len(other_datasets)} alternatives."
        )
    return other_datasets[0]


def _resolve_dataset_assets(
    paths_cfg: Dict[str, Any],
    dataset_name: str,
) -> Tuple[Path, Path, Dict[str, Any]]:
    dataset_lookup = _dataset_paths_lookup(paths_cfg)
    key = str(dataset_name).strip().lower()
    if key not in dataset_lookup:
        available = ", ".join(sorted(dataset_lookup))
        raise ValueError(
            f"Dataset '{dataset_name}' not found in Config.paths.dataset_paths. Available: {available}"
        )

    entry = dataset_lookup[key]
    data_dir_value = entry.get("data_dir")
    gold_labels_value = entry.get("gold_labels")
    if not data_dir_value or not gold_labels_value:
        raise ValueError(
            f"Dataset '{dataset_name}' must define both 'data_dir' and 'gold_labels' "
            "in Config.paths.dataset_paths."
        )
    return Path(str(data_dir_value)), Path(str(gold_labels_value)), entry


def _common_suffix_token_count(left: str, right: str) -> int:
    left_tokens = [token for token in re.split(r"[^A-Za-z0-9]+", left.lower()) if token]
    right_tokens = [token for token in re.split(r"[^A-Za-z0-9]+", right.lower()) if token]
    count = 0
    for left_token, right_token in zip(reversed(left_tokens), reversed(right_tokens)):
        if left_token != right_token:
            break
        count += 1
    return count


def _token_set(value: str) -> set[str]:
    return {token for token in re.split(r"[^A-Za-z0-9]+", value.lower()) if token}


def _resolve_dataset_base_predictions_path(
    *,
    paths_cfg: Dict[str, Any],
    dataset_name: str,
    models: Sequence[str],
    zero_cfg: Dict[str, Any],
    loo_cfg: Dict[str, Any],
    require_exists: bool = True,
) -> Path:
    _, _, dataset_entry = _resolve_dataset_assets(paths_cfg, dataset_name)

    explicit_base_predictions = dataset_entry.get("base_predictions_path")
    if explicit_base_predictions:
        path = Path(str(explicit_base_predictions))
        if require_exists and not path.exists():
            raise FileNotFoundError(
                f"Configured base_predictions_path for dataset '{dataset_name}' not found: {path}"
            )
        return path

    zero_subdir = str(zero_cfg.get("output_subdir", "zero_shot"))
    explicit_output_root = dataset_entry.get("output_root")
    if explicit_output_root:
        path = Path(str(explicit_output_root)) / zero_subdir / "all_models.pkl"
        if require_exists and not path.exists():
            raise FileNotFoundError(
                f"Configured output_root for dataset '{dataset_name}' does not contain "
                f"{zero_subdir}/all_models.pkl: {path}"
            )
        return path

    results_root = PROJECT_ROOT / "results"
    if not results_root.exists():
        raise FileNotFoundError(
            "Could not auto-discover cross-dataset zero-shot predictions because "
            f"'{results_root}' does not exist."
        )

    desired_data_dir = str(dataset_entry.get("data_dir", "")).strip()
    desired_gold_labels = str(dataset_entry.get("gold_labels", "")).strip()
    desired_models = [str(model_name) for model_name in models]
    desired_zero_prompt_option = zero_cfg.get("prompt_option")
    desired_loo_prompt_option = loo_cfg.get("prompt_option")
    current_output_root_name = Path(str(paths_cfg.get("output_root", ""))).name
    desired_data_dir_name = Path(desired_data_dir).name if desired_data_dir else ""
    desired_data_dir_tokens = _token_set(desired_data_dir_name)

    candidates: List[Dict[str, Any]] = []
    for snapshot_path in sorted(results_root.glob("*/config_snapshot.json")):
        try:
            snapshot_cfg = _load_json(snapshot_path)
        except Exception:
            continue

        snapshot_paths = snapshot_cfg.get("paths", {})
        snapshot_dataset = str(snapshot_paths.get("dataset", "")).strip().lower()
        if snapshot_dataset != str(dataset_name).strip().lower():
            continue

        predictions_path = snapshot_path.parent / zero_subdir / "all_models.pkl"
        if not predictions_path.exists():
            continue

        score = 0
        if desired_data_dir and str(snapshot_paths.get("data_dir", "")).strip() == desired_data_dir:
            score += 100
        if desired_gold_labels and str(snapshot_paths.get("gold_labels", "")).strip() == desired_gold_labels:
            score += 30

        snapshot_models = [str(model_name) for model_name in snapshot_cfg.get("models", [])]
        if snapshot_models == desired_models:
            score += 20
        elif set(snapshot_models) == set(desired_models):
            score += 10

        snapshot_zero_prompt_option = snapshot_cfg.get("zero_shot", {}).get("prompt_option")
        if desired_zero_prompt_option and snapshot_zero_prompt_option == desired_zero_prompt_option:
            score += 5

        snapshot_loo_prompt_option = snapshot_cfg.get("loo", {}).get("prompt_option")
        if desired_loo_prompt_option and snapshot_loo_prompt_option == desired_loo_prompt_option:
            score += 3

        snapshot_output_root_name = Path(
            str(snapshot_paths.get("output_root", snapshot_path.parent))
        ).name
        if current_output_root_name and snapshot_output_root_name:
            score += 7 * _common_suffix_token_count(
                current_output_root_name,
                snapshot_output_root_name,
            )
        if desired_data_dir_tokens and snapshot_output_root_name:
            score += 4 * len(
                desired_data_dir_tokens.intersection(_token_set(snapshot_output_root_name))
            )

        candidates.append(
            {
                "score": score,
                "snapshot_path": snapshot_path,
                "predictions_path": predictions_path,
            }
        )

    if not candidates:
        raise FileNotFoundError(
            "Could not auto-discover cross-dataset zero-shot predictions for "
            f"dataset '{dataset_name}'. Add dataset_paths.{dataset_name}.base_predictions_path "
            "or dataset_paths.<dataset>.output_root to the config."
        )

    candidates.sort(
        key=lambda item: (item["score"], str(item["predictions_path"])),
        reverse=True,
    )
    best = candidates[0]
    tied = [item for item in candidates if item["score"] == best["score"]]
    if len(tied) > 1:
        tied_paths = ", ".join(str(item["predictions_path"]) for item in tied[:5])
        raise ValueError(
            "Cross-dataset zero-shot prediction discovery is ambiguous for "
            f"dataset '{dataset_name}'. Matching candidates: {tied_paths}. "
            "Set dataset_paths.<dataset>.base_predictions_path explicitly."
        )

    return Path(best["predictions_path"])


def _resolve_device(device_cfg: Optional[str]) -> str:
    if device_cfg and device_cfg.lower() != "auto":
        return device_cfg
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_cached_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["file", "model", "raw_output"])
    df = pd.read_pickle(path)
    if "file" in df.columns:
        df = df.copy()
        df["file"] = df["file"].astype(str)
    return df


def _filter_base_predictions(
    base_predictions: pd.DataFrame,
    *,
    models: Sequence[str],
    doc_ids: Sequence[str],
) -> pd.DataFrame:
    if "file" not in base_predictions.columns or "model" not in base_predictions.columns:
        raise KeyError("Base predictions must contain columns 'file' and 'model'.")

    work = base_predictions.copy()
    work["file"] = work["file"].astype(str)
    work = work[work["model"].isin(models)]
    work = work[work["file"].isin(set(doc_ids))].reset_index(drop=True)
    if work.empty:
        raise RuntimeError(
            "No base predictions remain after filtering by configured models and documents."
        )
    return work


def _missing_files(cached_df: pd.DataFrame, doc_ids: Sequence[str]) -> set[str]:
    if cached_df.empty or "file" not in cached_df.columns:
        return set(doc_ids)
    cached_files = {str(fid) for fid in cached_df["file"].tolist() if fid and fid != "nan"}
    return set(doc_ids).difference(cached_files)


def _results_path(output_dir: Path, model_name: str) -> Path:
    return output_dir / f"results_{model_name.replace('/', '_')}.pkl"


def _canonical_gold_label(value: Any) -> Optional[str]:
    canon = canonicalize_label(value)
    if canon is not None:
        return canon
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().lower()
    return text or None


def _load_gold_labels_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        frame = pd.read_pickle(path)
    else:
        frame = pd.read_csv(path)

    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"Gold labels did not load as a DataFrame: {path}")
    if "file" not in frame.columns or "label" not in frame.columns:
        raise KeyError(f"Gold labels must contain 'file' and 'label': {path}")

    work = frame.copy()
    work["file"] = work["file"].astype(str)
    work = work.drop_duplicates(subset=["file"], keep="last").reset_index(drop=True)
    return work


def _compare_gold_labels(old_gold_path: Path, new_gold_path: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    old_df = _load_gold_labels_frame(old_gold_path)[["file", "label"]].rename(columns={"label": "label_old"})
    new_df = _load_gold_labels_frame(new_gold_path)[["file", "label"]].rename(columns={"label": "label_new"})

    merged = old_df.merge(new_df, on="file", how="outer", indicator=True)
    merged["label_old_canon"] = merged["label_old"].map(_canonical_gold_label)
    merged["label_new_canon"] = merged["label_new"].map(_canonical_gold_label)

    def _status(row: pd.Series) -> str:
        if row["_merge"] == "left_only":
            return "removed"
        if row["_merge"] == "right_only":
            return "added"
        if row["label_old_canon"] != row["label_new_canon"]:
            return "label_changed"
        return "unchanged"

    merged["change_type"] = merged.apply(_status, axis=1)
    changed = merged[merged["change_type"] != "unchanged"].copy().reset_index(drop=True)

    report = {
        "old_gold_labels": str(old_gold_path),
        "new_gold_labels": str(new_gold_path),
        "n_old_rows": int(len(old_df)),
        "n_new_rows": int(len(new_df)),
        "n_changed_total": int(len(changed)),
        "n_label_changed": int((merged["change_type"] == "label_changed").sum()),
        "n_added": int((merged["change_type"] == "added").sum()),
        "n_removed": int((merged["change_type"] == "removed").sum()),
        "changed_files": changed["file"].astype(str).tolist(),
    }
    return changed, report


def _resolve_rerun_output_root(
    *,
    source_output_root: Path,
    rerun_cfg: Dict[str, Any],
    cli_output_root: Optional[str],
) -> Path:
    if cli_output_root:
        return Path(cli_output_root)
    suffix = str(rerun_cfg.get("output_suffix", "_UPD")).strip() or "_UPD"
    return source_output_root.parent / f"{source_output_root.name}{suffix}"


def _rewrite_path_for_copied_root(
    value: Optional[str],
    *,
    source_output_root: Path,
    target_output_root: Path,
) -> Optional[Path]:
    if not value:
        return None

    raw = Path(str(value))
    candidate = raw if raw.is_absolute() else (PROJECT_ROOT / raw)
    try:
        rel = candidate.resolve().relative_to(source_output_root.resolve())
    except ValueError:
        return raw
    return target_output_root / rel


def _copy_results_tree(*, source_output_root: Path, target_output_root: Path, dry_run: bool) -> None:
    if dry_run:
        print(f"[rerun][dry-run] Would copy {source_output_root} -> {target_output_root}")
        return
    print(f"[rerun] Copying {source_output_root} -> {target_output_root}")
    target_output_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_output_root, target_output_root, dirs_exist_ok=True)
    print(f"[rerun] Copied {source_output_root} -> {target_output_root}")


def _remove_path_if_exists(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _parse_seed_k_dirname(dirname: str) -> Tuple[Optional[int], Optional[int]]:
    match = re.fullmatch(r"seed(?P<seed>\d+)_k(?P<k>\d+)", str(dirname).strip())
    if not match:
        return None, None
    return int(match.group("seed")), int(match.group("k"))


def _iter_loo_run_dirs(selection_dir: Path) -> List[Tuple[Path, str, int, int]]:
    runs: List[Tuple[Path, str, int, int]] = []
    if not selection_dir.exists():
        return runs

    for path in sorted(p for p in selection_dir.rglob("*") if p.is_dir()):
        seed, k = _parse_seed_k_dirname(path.name)
        if seed is None or k is None:
            continue
        rel = path.relative_to(selection_dir)
        composition = "all" if len(rel.parts) == 1 else str(rel.parts[0])
        runs.append((path, composition, seed, k))
    return runs


def _load_run_predictions(run_dir: Path, *, models: Sequence[str]) -> pd.DataFrame:
    result_files = sorted(run_dir.glob("results_*.pkl"))
    frames: List[pd.DataFrame] = []
    if result_files:
        for path in result_files:
            frames.append(_load_cached_dataframe(path))
    else:
        all_models_path = run_dir / "all_models.pkl"
        if all_models_path.exists():
            frames.append(_load_cached_dataframe(all_models_path))

    if not frames:
        return pd.DataFrame(columns=["file", "model", "raw_output"])

    combined = pd.concat(frames, ignore_index=True)
    if "model" in combined.columns:
        combined = combined[combined["model"].isin(models)].reset_index(drop=True)
    return combined


def _method_name_for_run(selection: str, composition: str, seed: int, k: int) -> str:
    suffix = f"seed{seed}_k{k}"
    if _is_ping_selection(selection):
        if composition == "all":
            return f"{selection}_{suffix}_loo"
        return f"{selection}_{composition}_{suffix}_loo"
    return f"{selection}_{suffix}_loo"


def _rebuild_loo_summaries_from_existing_runs(
    *,
    loo_output_root: Path,
    models: Sequence[str],
    eval_gold_df: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    loo_output_root.mkdir(parents=True, exist_ok=True)
    for stale in loo_output_root.glob("summary_*.csv"):
        stale.unlink()

    valid_selections = {"random", "random_ra", "ping", "ping_ra", "cross_ping", "cross_ping_ra"}
    by_selection_and_composition: Dict[Tuple[str, str], List[pd.DataFrame]] = {}

    for selection_dir in sorted(path for path in loo_output_root.iterdir() if path.is_dir()):
        selection = selection_dir.name
        if selection not in valid_selections:
            continue
        for run_dir, composition, seed, k in _iter_loo_run_dirs(selection_dir):
            combined = _load_run_predictions(run_dir, models=models)
            if combined.empty:
                continue
            summary, _ = evaluate_by_model(combined, eval_gold_df)
            if summary.empty:
                continue
            summary = summary.assign(
                selection=selection,
                composition=composition,
                method=_method_name_for_run(selection, composition, seed, k),
                seed=seed,
                k=k,
            )
            by_selection_and_composition.setdefault((selection, composition), []).append(summary)

    summary_frames: Dict[str, pd.DataFrame] = {}
    summary_all_by_composition: Dict[str, List[pd.DataFrame]] = {}
    for (selection, composition), frames in by_selection_and_composition.items():
        summary_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        summary_frames[f"{selection}::{composition}"] = summary_df
        if summary_df.empty:
            continue

        summary_all_by_composition.setdefault(composition, []).append(summary_df)
        summary_name = (
            f"summary_{selection}.csv"
            if composition == "all"
            else f"summary_{selection}_{composition}.csv"
        )
        summary_path = loo_output_root / summary_name
        summary_df.to_csv(summary_path, index=False)
        print(f"[loo] Wrote {summary_path}")

    for composition, frames in summary_all_by_composition.items():
        summary_all = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if summary_all.empty:
            continue
        summary_name = "summary_all.csv" if composition == "all" else f"summary_all_{composition}.csv"
        summary_path = loo_output_root / summary_name
        summary_all.to_csv(summary_path, index=False)
        print(f"[loo] Wrote {summary_path}")
        summary_frames[f"all::{composition}"] = summary_all

    return summary_frames


def _refresh_regex_baseline(
    *,
    config_snapshot_path: Path,
    output_dir: Path,
    dry_run: bool,
) -> None:
    cmd = [
        sys.executable,
        str((PROJECT_ROOT / "evaluate_mimic_regex_baseline.py").resolve()),
        "--config",
        str(config_snapshot_path),
        "--output-dir",
        str(output_dir),
    ]
    if dry_run:
        print(f"[rerun][dry-run] Would refresh regex baseline: {' '.join(cmd)}")
        return
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


def _write_rerun_report(output_root: Path, report: Dict[str, Any]) -> None:
    report_path = output_root / "rerun_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[rerun] Wrote {report_path}")


def _random_sample_docs(
    docs: List[Dict[str, Any]],
    *,
    sample_size: int,
    random_state: int,
) -> List[Dict[str, Any]]:
    if sample_size >= len(docs):
        return docs
    idx = (
        pd.Series(range(len(docs)))
        .sample(n=sample_size, random_state=random_state, replace=False)
        .sort_values()
        .tolist()
    )
    return [docs[i] for i in idx]


def _sample_documents(
    docs: List[Dict[str, Any]],
    *,
    sampling_cfg: Dict[str, Any],
    gold_labels_path: Optional[Path],
) -> List[Dict[str, Any]]:
    enabled = bool(sampling_cfg.get("enabled", False))
    fraction = float(sampling_cfg.get("fraction", 1.0))
    stratified = bool(sampling_cfg.get("stratified", True))
    random_state = int(sampling_cfg.get("random_state", 42))

    if not enabled or fraction >= 1.0:
        print(f"[sampling] Using all {len(docs)} documents (sampling disabled).")
        return docs

    if fraction <= 0.0 or fraction > 1.0:
        raise ValueError("document_sampling.fraction must be in (0, 1].")

    n_total = len(docs)
    if n_total == 0:
        return docs

    sample_size = max(1, int(round(n_total * fraction)))
    if sample_size >= n_total:
        print(f"[sampling] Requested sample size {sample_size} >= {n_total}; using all documents.")
        return docs

    if not stratified:
        sampled = _random_sample_docs(docs, sample_size=sample_size, random_state=random_state)
        print(f"[sampling] Random sample: {len(sampled)}/{n_total} docs (fraction={fraction}).")
        return sampled

    if gold_labels_path is None or not gold_labels_path.exists():
        raise ValueError(
            "Stratified sampling requires a valid Config.paths.gold_labels file."
        )

    gold_df = _load_gold_labels_frame(gold_labels_path)
    if "file" not in gold_df.columns or "label" not in gold_df.columns:
        raise KeyError("Gold labels must contain columns 'file' and 'label' for stratified sampling.")

    gold_keys_orig = [str(x) for x in gold_df["file"].dropna().tolist()]
    gold_keys_norm = [k.lower() for k in gold_keys_orig]
    gold_map = {
        str(f): str(l)
        for f, l in zip(gold_df["file"].tolist(), gold_df["label"].tolist())
    }

    labels: List[str] = []
    for doc in docs:
        file_id = str(doc.get("file", ""))
        file_norm = file_id.lower()
        best_key = None
        best_len = -1
        for orig, norm in zip(gold_keys_orig, gold_keys_norm):
            if norm and norm in file_norm and len(norm) > best_len:
                best_key = orig
                best_len = len(norm)
        labels.append(gold_map.get(best_key, "__unknown__"))

    n_classes = len(set(labels))
    class_counts = pd.Series(labels).value_counts()
    can_stratify = n_classes > 1 and (class_counts.min() >= 2) and (sample_size >= n_classes)

    if not can_stratify:
        sampled = _random_sample_docs(docs, sample_size=sample_size, random_state=random_state)
        print(
            "[sampling] Stratified sampling not feasible with current class counts; "
            f"used random sample {len(sampled)}/{n_total} (fraction={fraction})."
        )
        return sampled

    try:
        indices = list(range(n_total))
        picked, _ = train_test_split(
            indices,
            train_size=sample_size,
            stratify=labels,
            random_state=random_state,
        )
    except Exception:
        sampled = _random_sample_docs(docs, sample_size=sample_size, random_state=random_state)
        print(
            "[sampling] Stratified split failed unexpectedly; "
            f"used random sample {len(sampled)}/{n_total} (fraction={fraction})."
        )
        return sampled

    picked = sorted(picked)
    sampled = [docs[i] for i in picked]
    print(f"[sampling] Stratified sample: {len(sampled)}/{n_total} docs (fraction={fraction}).")
    return sampled


def _safe_model_tag(model_name: str) -> str:
    return model_name.replace("/", "_")


def _append_error_log(
    *,
    error_log_path: Path,
    stage: str,
    model_name: str,
    exc: Exception,
    context: Optional[Dict[str, Any]] = None,
) -> None:
    context = context or {}
    stamp = datetime.now(timezone.utc).isoformat()
    lines = [
        "=" * 80,
        f"time_utc: {stamp}",
        f"stage: {stage}",
        f"model: {model_name}",
    ]
    for key in sorted(context.keys()):
        lines.append(f"{key}: {context[key]}")
    lines.append(f"error_type: {type(exc).__name__}")
    lines.append(f"error_message: {exc}")
    lines.append("traceback:")
    lines.append(traceback.format_exc().rstrip())
    lines.append("")
    error_log_path.parent.mkdir(parents=True, exist_ok=True)
    with error_log_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _write_model_samples(
    *,
    df: pd.DataFrame,
    model_name: str,
    output_dir: Path,
    sample_count: int = 1,
    random_state: int = 0,
) -> None:
    if df is None or df.empty:
        return
    if sample_count <= 0:
        return

    work = df.copy()
    if "model" in work.columns:
        work = work[work["model"] == model_name].reset_index(drop=True)
    if work.empty:
        return

    n = min(sample_count, len(work))
    sampled = work.sample(n=n, random_state=random_state) if len(work) > n else work.head(n)

    path = output_dir / f"sample_{_safe_model_tag(model_name)}.txt"
    lines = [
        f"model: {model_name}",
        f"samples: {n}",
        "",
    ]
    for i, (_, row) in enumerate(sampled.iterrows(), start=1):
        file_id = row["file"] if "file" in row and pd.notna(row["file"]) else "N/A"
        prompt = row["prompt"] if "prompt" in row and pd.notna(row["prompt"]) else "[prompt not available]"
        output = row["raw_output"] if "raw_output" in row and pd.notna(row["raw_output"]) else "[raw_output not available]"
        lines.append(f"--- sample {i} ---")
        lines.append(f"file: {file_id}")
        lines.append("")
        lines.append("prompt:")
        lines.append(str(prompt))
        lines.append("")
        lines.append("raw_output:")
        lines.append(str(output))
        lines.append("")

    output_dir.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def _write_ping_selection_stats(
    *,
    builder: LeaveOneOutPromptBuilder,
    model_name: str,
    output_dir: Path,
) -> None:
    if not hasattr(builder, "ping_selection_stats_frames"):
        return
    summary_df, events_df = builder.ping_selection_stats_frames()
    if summary_df is None or summary_df.empty:
        return

    tag = _safe_model_tag(model_name)
    summary_path = output_dir / f"selection_stats_{tag}.csv"
    summary_df.to_csv(summary_path, index=False)

    if events_df is not None and not events_df.empty:
        events_path = output_dir / f"selection_events_{tag}.csv"
        events_df.to_csv(events_path, index=False)


def _run_zero_shot(
    *,
    docs: List[Dict[str, Any]],
    doc_ids: Sequence[str],
    models: Sequence[str],
    output_dir: Path,
    prompt_template: str,
    dry_run: bool,
    write_samples_txt: bool,
    sample_count_per_model: int,
    sample_random_state: int,
    write_error_log: bool,
) -> pd.DataFrame:
    _ensure_dir(output_dir)
    error_log_path = output_dir / "errors.log"

    if dry_run:
        print("[dry-run] Zero-shot stage")
        print(f"[dry-run] Output dir: {output_dir}")
        for model_name in models:
            print(f"[dry-run] Would run zero-shot for {model_name}")
        return pd.DataFrame(columns=["file", "model", "raw_output"])

    frames: List[pd.DataFrame] = []
    for model_name in models:
        result_path = _results_path(output_dir, model_name)
        cached_df = _load_cached_dataframe(result_path)
        missing = _missing_files(cached_df, doc_ids)

        if not missing and not cached_df.empty:
            print(f"[zero-shot] Skipping {model_name}; cached results detected ({len(cached_df)} rows).")
            frames.append(cached_df.copy())
            if write_samples_txt:
                _write_model_samples(
                    df=cached_df,
                    model_name=model_name,
                    output_dir=output_dir,
                    sample_count=sample_count_per_model,
                    random_state=sample_random_state,
                )
            continue

        print(f"[zero-shot] Running {model_name}")
        try:
            df = run_zero_shot_classification(
                docs,
                model_name,
                results_dir=str(output_dir),
                PROMPT_TEMPLATE=prompt_template,
            )
        except Exception as exc:
            if write_error_log:
                _append_error_log(
                    error_log_path=error_log_path,
                    stage="zero-shot",
                    model_name=model_name,
                    exc=exc,
                    context={"output_dir": str(output_dir)},
                )
            print(f"[zero-shot] ERROR for {model_name}: {exc}")
            continue

        frames.append(df)
        if write_samples_txt:
            _write_model_samples(
                df=df,
                model_name=model_name,
                output_dir=output_dir,
                sample_count=sample_count_per_model,
                random_state=sample_random_state,
            )
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["file", "model", "raw_output"])
    combined.to_pickle(output_dir / "all_models.pkl")
    print(f"[zero-shot] Wrote {output_dir / 'all_models.pkl'} ({len(combined)} rows)")
    return combined


def _selection_prompt_kwargs(loo_cfg: Dict[str, Any], selection: str) -> Dict[str, Any]:
    shared = dict(loo_cfg.get("prompt_kwargs", {}))
    by_selection = loo_cfg.get("prompt_kwargs_by_selection", {})
    specific: Dict[str, Any] = {}
    if _is_ping_selection(selection):
        specific.update(dict(by_selection.get("ping", {})))
    elif _is_random_selection(selection):
        specific.update(dict(by_selection.get("random", {})))
    specific.update(dict(by_selection.get(selection, {})))
    shared.update(specific)
    return shared


def _selection_overrides(loo_cfg: Dict[str, Any], selection: str) -> Dict[str, Any]:
    overrides_by_selection = loo_cfg.get("selection_overrides", {})
    merged: Dict[str, Any] = {}
    if _is_ping_selection(selection):
        merged.update(dict(overrides_by_selection.get("ping", {})))
    elif _is_random_selection(selection):
        merged.update(dict(overrides_by_selection.get("random", {})))
    merged.update(dict(overrides_by_selection.get(selection, {})))
    return merged


def _selection_uses_random_pong(loo_cfg: Dict[str, Any], selection: str) -> bool:
    if not _is_ping_selection(selection):
        return True
    ping_overrides = _selection_overrides(loo_cfg, selection)
    ping_use_random_pong_raw = ping_overrides.get("use_random_pong", True)
    if isinstance(ping_use_random_pong_raw, str):
        return ping_use_random_pong_raw.strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
    return bool(ping_use_random_pong_raw)


def _prepare_builder(
    *,
    resources,
    selection: str,
    selection_kwargs: Dict[str, Any],
    query_template: str,
    prompt_kwargs: Dict[str, Any],
    exclude_target_from_pool: bool = True,
) -> LeaveOneOutPromptBuilder:
    return LeaveOneOutPromptBuilder(
        resources,
        selection=selection,
        selection_kwargs=selection_kwargs,
        query_template=query_template,
        cache_prompts=True,
        prompt_kwargs=prompt_kwargs,
        exclude_target_from_pool=exclude_target_from_pool,
    )


def _parse_ping_composition(raw_value: Optional[str], default_size: int) -> Tuple[str, int]:
    if default_size <= 0:
        raise ValueError("LOO ping committee size must be >= 1.")

    value = (raw_value or DEFAULT_PING_COMPOSITION).strip().lower()
    normalized = re.sub(r"[\s\-_]+", "", value)
    if normalized == "all":
        return "all", default_size

    aliases = {
        "hi": "hi",
        "high": "hi",
        "lo": "lo",
        "low": "lo",
        "mix": "mix",
    }
    if normalized in aliases:
        return aliases[normalized], default_size

    match = re.fullmatch(r"(hi|lo|mix)(\d+)", normalized)
    if match:
        mode = match.group(1)
        size = int(match.group(2))
        if size <= 0:
            raise ValueError("Ping composition size must be >= 1.")
        return mode, size

    raise ValueError(
        "Unsupported ping composition. Use one of: "
        "'all', 'hi', 'lo', 'mix', 'hi3', 'lo3', 'mix3'."
    )


def _ping_composition_tag(mode: str, committee_size: int) -> str:
    if mode == "all":
        return "all"
    return f"{mode}{committee_size}"


def _build_zero_shot_ranking(
    base_predictions: pd.DataFrame,
    gold_df: pd.DataFrame,
    models: Sequence[str],
) -> Tuple[List[str], Dict[str, float]]:
    summary, _ = evaluate_by_model(base_predictions, gold_df)
    score_map: Dict[str, float] = {}
    for _, row in summary.iterrows():
        if pd.isna(row.get("f1")):
            continue
        model_name = str(row.get("model"))
        score_map[model_name] = float(row.get("f1"))

    def sort_key(model_name: str) -> Tuple[int, float, str]:
        score = score_map.get(model_name)
        if score is None or pd.isna(score):
            return (1, 0.0, model_name)
        return (0, -float(score), model_name)

    ranked = sorted(models, key=sort_key)
    return ranked, score_map


def _select_committee_from_ranked(
    ranked_remaining: Sequence[str],
    *,
    composition_mode: str,
    committee_size: int,
) -> List[str]:
    if not ranked_remaining:
        return []

    k = min(max(1, committee_size), len(ranked_remaining))
    ranked = list(ranked_remaining)

    if composition_mode == "all":
        return ranked
    if composition_mode == "hi":
        return ranked[:k]
    if composition_mode == "lo":
        return ranked[-k:]
    if composition_mode == "mix":
        # Best, median, worst (and fill deterministically if k > 3).
        anchor_indices = [0, len(ranked) // 2, len(ranked) - 1]
        picked: List[str] = []
        for idx in anchor_indices:
            candidate = ranked[idx]
            if candidate not in picked:
                picked.append(candidate)
            if len(picked) >= k:
                break
        if len(picked) < k:
            for candidate in ranked:
                if candidate not in picked:
                    picked.append(candidate)
                if len(picked) >= k:
                    break
        return picked[:k]
    raise ValueError(f"Unknown composition mode: {composition_mode}")


def _build_ping_committees(
    *,
    models: Sequence[str],
    ranked_models: Sequence[str],
    composition_mode: str,
    committee_size: int,
) -> Dict[str, List[str]]:
    committees: Dict[str, List[str]] = {}
    ranked_models_list = list(ranked_models)
    for target_model in models:
        ranked_remaining = [m for m in ranked_models_list if m != target_model]
        committee = _select_committee_from_ranked(
            ranked_remaining,
            composition_mode=composition_mode,
            committee_size=committee_size,
        )
        committees[target_model] = committee
    return committees


def _run_loo_config(
    *,
    docs: List[Dict[str, Any]],
    doc_ids: Sequence[str],
    models: Sequence[str],
    selection: str,
    seed: int,
    k: int,
    output_root: Path,
    base_predictions: pd.DataFrame,
    selector_gold_df: pd.DataFrame,
    eval_gold_df: pd.DataFrame,
    shared_resources,
    loo_cfg: Dict[str, Any],
    dry_run: bool,
    ping_resource_cache: Dict[Any, Any],
    ping_composition_mode: str,
    ping_committee_size: int,
    ping_committees: Optional[Dict[str, List[str]]],
    write_samples_txt: bool,
    sample_count_per_model: int,
    sample_random_state: int,
    write_error_log: bool,
    exclude_target_from_pool: bool = True,
) -> Dict[str, Any]:
    suffix = f"seed{seed}_k{k}"
    ping_composition_tag = _ping_composition_tag(ping_composition_mode, ping_committee_size)
    ping_like = _is_ping_selection(selection)
    if ping_like:
        if ping_composition_tag == "all":
            method_name = f"{selection}_{suffix}_loo"
            output_dir = _ensure_dir(output_root / selection / suffix)
        else:
            method_name = f"{selection}_{ping_composition_tag}_{suffix}_loo"
            output_dir = _ensure_dir(output_root / selection / ping_composition_tag / suffix)
    else:
        method_name = f"{selection}_{suffix}_loo"
        output_dir = _ensure_dir(output_root / selection / suffix)
    error_log_path = output_dir / "errors.log"

    selection_kwargs = {"k": k, "seed": seed}
    selection_kwargs.update(_selection_overrides(loo_cfg, selection))
    if _is_random_selection(selection):
        # Random LOO now defaults to reveal-based pool drawing:
        # draw from unlabeled pool, reveal gold, and keep draws that fill class quotas.
        selection_kwargs.setdefault("simulate_reveal", True)
    prompt_kwargs = _selection_prompt_kwargs(loo_cfg, selection)
    query_template = loo_cfg.get("query_template", DEFAULT_QUERY_TEMPLATE)

    if dry_run:
        print(f"[dry-run] {method_name}")
        print(f"[dry-run] selection kwargs: {selection_kwargs}")
        print(f"[dry-run] output dir: {output_dir}")
        return {
            "method": method_name,
            "selection": selection,
            "selection_kwargs": selection_kwargs,
            "output_dir": output_dir,
            "combined": pd.DataFrame(columns=["file", "model", "raw_output"]),
            "summary": pd.DataFrame(),
            "merged": pd.DataFrame(),
        }

    cache_records: Dict[str, Dict[str, Any]] = {}
    for model_name in models:
        result_path = _results_path(output_dir, model_name)
        cached_df = _load_cached_dataframe(result_path)
        missing = _missing_files(cached_df, doc_ids)
        cache_records[model_name] = {"dataframe": cached_df, "missing": missing}

    frames: List[pd.DataFrame] = []
    shared_builder: Optional[LeaveOneOutPromptBuilder] = None

    for model_name in models:
        cached_df = cache_records[model_name]["dataframe"]
        missing = cache_records[model_name]["missing"]

        has_full_cache = not missing and not cached_df.empty
        if has_full_cache:
            print(f"[{method_name}] Skipping {model_name}; cached results detected ({len(cached_df)} rows).")
            frames.append(cached_df.copy())
            if write_samples_txt:
                _write_model_samples(
                    df=cached_df,
                    model_name=model_name,
                    output_dir=output_dir,
                    sample_count=sample_count_per_model,
                    random_state=sample_random_state,
                )
            continue

        try:
            if ping_like:
                if ping_committees is not None:
                    committee_models = list(ping_committees.get(model_name, []))
                else:
                    committee_models = [m for m in models if m != model_name]
                if not committee_models:
                    raise RuntimeError(
                        f"No committee models available for {selection} selection (target='{model_name}')."
                    )
                committee_preds = base_predictions[base_predictions["model"].isin(committee_models)]

                cache_key = (ping_composition_tag, model_name, tuple(committee_models))
                if cache_key not in ping_resource_cache:
                    compute_embeds = shared_resources.embedding_cache is None
                    ping_resource_cache[cache_key] = prepare_loo_resources(
                        committee_preds,
                        docs=None,
                        text_lookup=shared_resources.text_lookup,
                        include_models=committee_models,
                        gold_df=selector_gold_df,
                        existing_embedding_cache=shared_resources.embedding_cache,
                        compute_embeddings=compute_embeds,
                        embed_model=loo_cfg.get("embed_model", "intfloat/multilingual-e5-large"),
                        embed_device=loo_cfg.get("embed_device"),
                        embed_prefix=loo_cfg.get("embed_prefix", "passage: "),
                    )
                    if shared_resources.embedding_cache is None:
                        shared_resources.embedding_cache = ping_resource_cache[cache_key].embedding_cache

                model_resources = ping_resource_cache[cache_key]
                builder = _prepare_builder(
                    resources=model_resources,
                    selection=selection,
                    selection_kwargs=selection_kwargs,
                    query_template=query_template,
                    prompt_kwargs=prompt_kwargs,
                    exclude_target_from_pool=exclude_target_from_pool,
                )
                builder.precompute_all()
            else:
                if shared_builder is None:
                    print(f"[{method_name}] Precomputing prompts...")
                    shared_builder = _prepare_builder(
                        resources=shared_resources,
                        selection=selection,
                        selection_kwargs=selection_kwargs,
                        query_template=query_template,
                        prompt_kwargs=prompt_kwargs,
                        exclude_target_from_pool=exclude_target_from_pool,
                    )
                    shared_builder.precompute_all()
                builder = shared_builder

            print(f"[{method_name}] Running {model_name}")
            df = run_zero_shot_classification(
                docs,
                model_name,
                results_dir=str(output_dir),
                PROMPT_TEMPLATE=builder,
            )
        except Exception as exc:
            if write_error_log:
                _append_error_log(
                    error_log_path=error_log_path,
                    stage=method_name,
                    model_name=model_name,
                    exc=exc,
                    context={
                        "output_dir": str(output_dir),
                        "selection": selection,
                        "composition": ping_composition_tag if ping_like else "all",
                        "seed": seed,
                        "k": k,
                    },
                )
            print(f"[{method_name}] ERROR for {model_name}: {exc}")
            continue

        frames.append(df)
        if write_samples_txt:
            _write_model_samples(
                df=df,
                model_name=model_name,
                output_dir=output_dir,
            sample_count=sample_count_per_model,
            random_state=sample_random_state,
        )
        if ping_like:
            _write_ping_selection_stats(
                builder=builder,
                model_name=model_name,
                output_dir=output_dir,
            )

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["file", "model", "raw_output"])
    combined.to_pickle(output_dir / "all_models.pkl")

    if combined.empty:
        summary = pd.DataFrame(
            columns=["model", "precision", "recall", "f1", "n", "selection", "composition", "method", "seed", "k"]
        )
        merged = pd.DataFrame()
    else:
        try:
            summary, merged = evaluate_by_model(combined, eval_gold_df)
            summary = summary.assign(
                selection=selection,
                composition=ping_composition_tag if ping_like else "all",
                method=method_name,
                seed=seed,
                k=k,
            )
        except Exception as exc:
            if write_error_log:
                _append_error_log(
                    error_log_path=error_log_path,
                    stage=f"{method_name}/evaluation",
                    model_name="ALL_MODELS",
                    exc=exc,
                    context={
                        "output_dir": str(output_dir),
                        "selection": selection,
                        "composition": ping_composition_tag if ping_like else "all",
                        "seed": seed,
                        "k": k,
                    },
                )
            print(f"[{method_name}] ERROR during evaluation: {exc}")
            summary = pd.DataFrame(
                columns=["model", "precision", "recall", "f1", "n", "selection", "composition", "method", "seed", "k"]
            )
            merged = pd.DataFrame()

    return {
        "method": method_name,
        "selection": selection,
        "selection_kwargs": selection_kwargs,
        "output_dir": output_dir,
        "combined": combined,
        "summary": summary,
        "merged": merged,
    }


def _validate_selection_names(selections: Sequence[str]) -> None:
    valid = {"random", "random_ra", "ping", "ping_ra", "cross_ping", "cross_ping_ra"}
    bad = [s for s in selections if s not in valid]
    if bad:
        raise ValueError(f"Unsupported selections: {bad}. Allowed: {sorted(valid)}")


def _run_loo(
    *,
    docs: List[Dict[str, Any]],
    doc_ids: Sequence[str],
    models: Sequence[str],
    output_root: Path,
    base_predictions_path: Path,
    gold_labels_path: Path,
    paths_cfg: Dict[str, Any],
    zero_cfg: Dict[str, Any],
    loo_cfg: Dict[str, Any],
    dry_run: bool,
    write_samples_txt: bool,
    sample_count_per_model: int,
    sample_random_state: int,
    write_error_log: bool,
) -> Dict[str, pd.DataFrame]:
    selections = loo_cfg.get("run_selections", ["random", "ping"])
    _validate_selection_names(selections)

    seeds = [int(s) for s in loo_cfg.get("seeds", [])]
    k_values = [int(k) for k in loo_cfg.get("k_values", [])]
    if not seeds or not k_values:
        raise ValueError("LOO config must provide non-empty 'seeds' and 'k_values'.")

    ping_composition_raw = loo_cfg.get("ping_composition", DEFAULT_PING_COMPOSITION)
    ping_committee_size_cfg = int(loo_cfg.get("ping_committee_size", DEFAULT_PING_COMMITTEE_SIZE))
    ping_composition_mode, ping_committee_size = _parse_ping_composition(
        ping_composition_raw,
        ping_committee_size_cfg,
    )
    ping_composition_tag = _ping_composition_tag(ping_composition_mode, ping_committee_size)
    any_ping_requested = any(_is_ping_selection(selection) for selection in selections)
    same_dataset_ping_requested = any(
        _is_ping_selection(selection) and not _is_cross_ping_selection(selection)
        for selection in selections
    )
    cross_ping_requested = any(_is_cross_ping_selection(selection) for selection in selections)
    same_dataset_required = any(not _is_cross_ping_selection(selection) for selection in selections)
    cross_dataset_name = _resolve_cross_dataset_name(paths_cfg) if cross_ping_requested else None

    def _selection_seed_list(selection: str) -> List[int]:
        if _is_ping_selection(selection) and not _selection_uses_random_pong(loo_cfg, selection):
            return [seeds[0]]
        return seeds

    if dry_run:
        print("[dry-run] LOO stage")
        if same_dataset_required:
            print(f"[dry-run] Base predictions: {base_predictions_path}")
        else:
            print("[dry-run] Base predictions: not used by current selections")
        print(f"[dry-run] Gold labels: {gold_labels_path}")
        print(f"[dry-run] selections={selections}, seeds={seeds}, k_values={k_values}")
        if any_ping_requested:
            print(
                "[dry-run] ping composition="
                f"{ping_composition_tag} (mode={ping_composition_mode}, committee_size={ping_committee_size})"
            )
        if cross_ping_requested and cross_dataset_name is not None:
            cross_data_dir, cross_gold_labels_path, _ = _resolve_dataset_assets(paths_cfg, cross_dataset_name)
            cross_base_predictions_path = _resolve_dataset_base_predictions_path(
                paths_cfg=paths_cfg,
                dataset_name=cross_dataset_name,
                models=models,
                zero_cfg=zero_cfg,
                loo_cfg=loo_cfg,
                require_exists=False,
            )
            print(
                f"[dry-run] cross-ping source dataset={cross_dataset_name}, "
                f"data_dir={cross_data_dir}, gold_labels={cross_gold_labels_path}, "
                f"base_predictions={cross_base_predictions_path}"
            )
        for selection in selections:
            selection_seeds = _selection_seed_list(selection)
            if _is_ping_selection(selection) and len(selection_seeds) == 1 and len(seeds) > 1:
                print(
                    f"[dry-run] {selection} uses deterministic no-pong mode; "
                    f"running one seed only (seed={selection_seeds[0]})."
                )
            for seed, k in product(selection_seeds, k_values):
                suffix = f"seed{seed}_k{k}"
                if _is_ping_selection(selection) and ping_composition_tag != "all":
                    print(f"[dry-run] Would run {selection}/{ping_composition_tag}/{suffix}")
                else:
                    print(f"[dry-run] Would run {selection}/{suffix}")
        return {
            "summary_random": pd.DataFrame(),
            "summary_random_ra": pd.DataFrame(),
            "summary_ping": pd.DataFrame(),
            "summary_ping_ra": pd.DataFrame(),
            "summary_all": pd.DataFrame(),
        }

    if same_dataset_required and not base_predictions_path.exists():
        raise FileNotFoundError(f"Base predictions not found: {base_predictions_path}")
    if not gold_labels_path.exists():
        raise FileNotFoundError(f"Gold labels not found: {gold_labels_path}")

    eval_gold_df = _load_gold_labels_frame(gold_labels_path)

    base_predictions = pd.DataFrame(columns=["file", "model", "raw_output"])
    resources = None
    ping_committees: Optional[Dict[str, List[str]]] = None
    ping_resource_cache: Dict[Any, Any] = {}
    if same_dataset_required:
        if not base_predictions_path.exists():
            raise FileNotFoundError(f"Base predictions not found: {base_predictions_path}")

        base_predictions = _filter_base_predictions(
            pd.read_pickle(base_predictions_path),
            models=models,
            doc_ids=doc_ids,
        )

        if same_dataset_ping_requested and ping_composition_mode != "all":
            ranked_models, score_map = _build_zero_shot_ranking(base_predictions, eval_gold_df, models)
            ping_committees = _build_ping_committees(
                models=models,
                ranked_models=ranked_models,
                composition_mode=ping_composition_mode,
                committee_size=ping_committee_size,
            )
            print(
                "[loo] Ping committee composition: "
                f"{ping_composition_tag} (mode={ping_composition_mode}, committee_size={ping_committee_size})"
            )
            print("[loo] Zero-shot ranking (best->worst):")
            for rank, model_name in enumerate(ranked_models, start=1):
                score = score_map.get(model_name)
                score_text = f"{score:.4f}" if score is not None and not pd.isna(score) else "N/A"
                print(f"  {rank:>2}. {model_name} (macro-F1={score_text})")
            print("[loo] Ping committees by target model:")
            for target_model in models:
                committee_models = ping_committees.get(target_model, [])
                print(f"  - {target_model}: {committee_models}")

        resources = prepare_loo_resources(
            base_predictions,
            docs=docs,
            include_models=models,
            gold_df=eval_gold_df,
            embed_model=loo_cfg.get("embed_model", "intfloat/multilingual-e5-large"),
            embed_device=loo_cfg.get("embed_device"),
            embed_prefix=loo_cfg.get("embed_prefix", "passage: "),
        )

    cross_base_predictions = pd.DataFrame(columns=["file", "model", "raw_output"])
    cross_selector_gold_df: Optional[pd.DataFrame] = None
    cross_resources = None
    cross_ping_committees: Optional[Dict[str, List[str]]] = None
    cross_ping_resource_cache: Dict[Any, Any] = {}
    if cross_ping_requested:
        if cross_dataset_name is None:
            raise RuntimeError("Failed to resolve the opposite dataset for cross-ping selection.")

        cross_data_dir, cross_gold_labels_path, _ = _resolve_dataset_assets(paths_cfg, cross_dataset_name)
        cross_base_predictions_path = _resolve_dataset_base_predictions_path(
            paths_cfg=paths_cfg,
            dataset_name=cross_dataset_name,
            models=models,
            zero_cfg=zero_cfg,
            loo_cfg=loo_cfg,
        )
        if not cross_data_dir.exists():
            raise FileNotFoundError(
                f"Cross-dataset data directory not found for '{cross_dataset_name}': {cross_data_dir}"
            )
        if not cross_gold_labels_path.exists():
            raise FileNotFoundError(
                f"Cross-dataset gold labels not found for '{cross_dataset_name}': {cross_gold_labels_path}"
            )
        if not cross_base_predictions_path.exists():
            raise FileNotFoundError(
                "Cross-dataset base predictions not found for "
                f"'{cross_dataset_name}': {cross_base_predictions_path}"
            )

        cross_docs = load_documents(str(cross_data_dir))
        if not cross_docs:
            raise RuntimeError(f"No documents found in cross-dataset source {cross_data_dir}")
        cross_doc_ids = [str(doc["file"]) for doc in cross_docs]

        cross_base_predictions = _filter_base_predictions(
            pd.read_pickle(cross_base_predictions_path),
            models=models,
            doc_ids=cross_doc_ids,
        )
        cross_selector_gold_df = _load_gold_labels_frame(cross_gold_labels_path)
        print(
            f"[loo] Cross-ping source dataset: {cross_dataset_name} "
            f"(docs={len(cross_docs)}, base_predictions={cross_base_predictions_path})"
        )

        if ping_composition_mode != "all":
            ranked_models, score_map = _build_zero_shot_ranking(
                cross_base_predictions,
                cross_selector_gold_df,
                models,
            )
            cross_ping_committees = _build_ping_committees(
                models=models,
                ranked_models=ranked_models,
                composition_mode=ping_composition_mode,
                committee_size=ping_committee_size,
            )
            print(
                "[loo] Cross-ping committee composition: "
                f"{ping_composition_tag} (mode={ping_composition_mode}, committee_size={ping_committee_size})"
            )
            print("[loo] Cross-ping zero-shot ranking (best->worst):")
            for rank, model_name in enumerate(ranked_models, start=1):
                score = score_map.get(model_name)
                score_text = f"{score:.4f}" if score is not None and not pd.isna(score) else "N/A"
                print(f"  {rank:>2}. {model_name} (macro-F1={score_text})")
            print("[loo] Cross-ping committees by target model:")
            for target_model in models:
                committee_models = cross_ping_committees.get(target_model, [])
                print(f"  - {target_model}: {committee_models}")

        cross_resources = prepare_loo_resources(
            cross_base_predictions,
            docs=cross_docs,
            include_models=models,
            gold_df=cross_selector_gold_df,
            embed_model=loo_cfg.get("embed_model", "intfloat/multilingual-e5-large"),
            embed_device=loo_cfg.get("embed_device"),
            embed_prefix=loo_cfg.get("embed_prefix", "passage: "),
        )

    summaries_by_selection: Dict[str, List[pd.DataFrame]] = {
        selection: [] for selection in selections
    }

    for selection in selections:
        selection_seeds = _selection_seed_list(selection)
        if _is_ping_selection(selection) and len(selection_seeds) == 1 and len(seeds) > 1:
            print(
                f"[loo] {selection} uses deterministic no-pong mode; "
                f"running one seed only (seed={selection_seeds[0]})."
            )
        selection_is_cross = _is_cross_ping_selection(selection)
        selection_base_predictions = cross_base_predictions if selection_is_cross else base_predictions
        selection_selector_gold_df = cross_selector_gold_df if selection_is_cross else eval_gold_df
        selection_resources = cross_resources if selection_is_cross else resources
        selection_ping_resource_cache = (
            cross_ping_resource_cache if selection_is_cross else ping_resource_cache
        )
        selection_ping_committees = (
            cross_ping_committees if selection_is_cross else ping_committees
        )

        if selection_resources is None:
            raise RuntimeError(f"No selector resources prepared for selection '{selection}'.")
        if selection_selector_gold_df is None:
            raise RuntimeError(f"No selector gold labels prepared for selection '{selection}'.")

        for seed, k in product(selection_seeds, k_values):
            result = _run_loo_config(
                docs=docs,
                doc_ids=doc_ids,
                models=models,
                selection=selection,
                seed=seed,
                k=k,
                output_root=output_root,
                base_predictions=selection_base_predictions,
                selector_gold_df=selection_selector_gold_df,
                eval_gold_df=eval_gold_df,
                shared_resources=selection_resources,
                loo_cfg=loo_cfg,
                dry_run=dry_run,
                ping_resource_cache=selection_ping_resource_cache,
                ping_composition_mode=ping_composition_mode,
                ping_committee_size=ping_committee_size,
                ping_committees=selection_ping_committees,
                write_samples_txt=write_samples_txt,
                sample_count_per_model=sample_count_per_model,
                sample_random_state=sample_random_state,
                write_error_log=write_error_log,
                exclude_target_from_pool=not selection_is_cross,
            )
            summaries_by_selection.setdefault(selection, []).append(result["summary"])

    summary_frames: Dict[str, pd.DataFrame] = {}
    for selection, frames in summaries_by_selection.items():
        summary_frames[selection] = (
            pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        )

    summary_random = summary_frames.get("random", pd.DataFrame())
    summary_random_ra = summary_frames.get("random_ra", pd.DataFrame())
    summary_ping = summary_frames.get("ping", pd.DataFrame())
    summary_ping_ra = summary_frames.get("ping_ra", pd.DataFrame())
    summary_cross_ping = summary_frames.get("cross_ping", pd.DataFrame())
    summary_cross_ping_ra = summary_frames.get("cross_ping_ra", pd.DataFrame())

    frames_for_all = [df for df in summary_frames.values() if not df.empty]
    summary_all = pd.concat(frames_for_all, ignore_index=True) if frames_for_all else pd.DataFrame()

    for selection, summary_df in summary_frames.items():
        if summary_df.empty:
            continue
        if _is_ping_selection(selection) and ping_composition_tag != "all":
            summary_path = output_root / f"summary_{selection}_{ping_composition_tag}.csv"
        else:
            summary_path = output_root / f"summary_{selection}.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"[loo] Wrote {summary_path}")
    if not summary_all.empty:
        if any_ping_requested and ping_composition_tag != "all":
            summary_all_path = output_root / f"summary_all_{ping_composition_tag}.csv"
        else:
            summary_all_path = output_root / "summary_all.csv"
        summary_all.to_csv(summary_all_path, index=False)
        print(f"[loo] Wrote {summary_all_path}")

    return {
        "summary_random": summary_random,
        "summary_random_ra": summary_random_ra,
        "summary_ping": summary_ping,
        "summary_ping_ra": summary_ping_ra,
        "summary_cross_ping": summary_cross_ping,
        "summary_cross_ping_ra": summary_cross_ping_ra,
        "summary_all": summary_all,
    }


def _run_gold_label_rerun(
    *,
    config_path: Path,
    effective_cfg: Dict[str, Any],
    rerun_cfg: Dict[str, Any],
    paths_cfg: Dict[str, Any],
    docs: List[Dict[str, Any]],
    doc_ids: Sequence[str],
    models: Sequence[str],
    gold_labels_path: Path,
    zero_cfg: Dict[str, Any],
    loo_cfg: Dict[str, Any],
    artifacts_cfg: Dict[str, Any],
    sampling_cfg: Dict[str, Any],
    cli_output_root: Optional[str],
    cli_base_predictions_path: Optional[str],
    dry_run: bool,
) -> None:
    source_output_root_value = rerun_cfg.get("source_output_root", paths_cfg.get("output_root"))
    if not source_output_root_value:
        raise ValueError("rerun.source_output_root or paths.output_root is required in rerun mode.")

    force_same_dataset_loo = bool(rerun_cfg.get("force_same_dataset_loo", False))
    force_cross_dataset_loo = bool(rerun_cfg.get("force_cross_dataset_loo", False))

    source_output_root = Path(str(source_output_root_value))
    if not source_output_root.exists():
        raise FileNotFoundError(f"Rerun source output root not found: {source_output_root}")

    target_output_root = _resolve_rerun_output_root(
        source_output_root=source_output_root,
        rerun_cfg=rerun_cfg,
        cli_output_root=cli_output_root,
    )
    if source_output_root.resolve() == target_output_root.resolve():
        raise ValueError("Rerun target output root must differ from the source output root.")

    old_gold_value = rerun_cfg.get("old_gold_labels")
    if not old_gold_value:
        source_snapshot_path = source_output_root / "config_snapshot.json"
        if source_snapshot_path.exists():
            source_snapshot = _load_json(source_snapshot_path)
            old_gold_value = source_snapshot.get("paths", {}).get("gold_labels")

    old_gold_path: Optional[Path] = None
    comparison_available = bool(old_gold_value)
    if comparison_available:
        old_gold_path = Path(str(old_gold_value))
        if not old_gold_path.exists():
            if force_same_dataset_loo:
                comparison_available = False
                old_gold_path = None
            else:
                raise FileNotFoundError(f"Old gold labels file not found: {old_gold_path}")
        elif force_same_dataset_loo and old_gold_path.resolve() == gold_labels_path.resolve():
            comparison_available = False
            old_gold_path = None
    elif not force_same_dataset_loo:
        raise ValueError(
            "Rerun mode requires rerun.old_gold_labels, or a source config_snapshot.json "
            "whose paths.gold_labels points to the old labels file. "
            "If that previous gold snapshot is unavailable, set rerun.force_same_dataset_loo=true."
        )
    if not gold_labels_path.exists():
        raise FileNotFoundError(f"New gold labels file not found: {gold_labels_path}")

    if comparison_available and old_gold_path is not None:
        changed_df, report = _compare_gold_labels(old_gold_path, gold_labels_path)
        report.update(
            {
                "dataset": str(paths_cfg.get("dataset", "mimic")),
                "source_output_root": str(source_output_root),
                "target_output_root": str(target_output_root),
                "rerun_time_utc": datetime.now(timezone.utc).isoformat(),
                "comparison_available": True,
                "force_same_dataset_loo": force_same_dataset_loo,
                "force_cross_dataset_loo": force_cross_dataset_loo,
            }
        )

        print("[rerun] Gold label comparison")
        print(f"[rerun] old_gold={old_gold_path}")
        print(f"[rerun] new_gold={gold_labels_path}")
        print(
            "[rerun] changed_total={n_changed_total} "
            "(label_changed={n_label_changed}, added={n_added}, removed={n_removed})".format(**report)
        )
        if report["changed_files"]:
            preview = ", ".join(report["changed_files"][:10])
            if len(report["changed_files"]) > 10:
                preview += ", ..."
            print(f"[rerun] changed_files_preview={preview}")
    else:
        changed_df = pd.DataFrame(columns=["file", "label_old", "label_new", "change_type"])
        report = {
            "old_gold_labels": None,
            "new_gold_labels": str(gold_labels_path),
            "n_old_rows": None,
            "n_new_rows": int(len(_load_gold_labels_frame(gold_labels_path))),
            "n_changed_total": None,
            "n_label_changed": None,
            "n_added": None,
            "n_removed": None,
            "changed_files": [],
            "dataset": str(paths_cfg.get("dataset", "mimic")),
            "source_output_root": str(source_output_root),
            "target_output_root": str(target_output_root),
            "rerun_time_utc": datetime.now(timezone.utc).isoformat(),
            "comparison_available": False,
            "force_same_dataset_loo": force_same_dataset_loo,
            "force_cross_dataset_loo": force_cross_dataset_loo,
        }
        print("[rerun] Gold label comparison unavailable; old gold snapshot not found.")
        print(f"[rerun] new_gold={gold_labels_path}")
        if force_same_dataset_loo:
            print("[rerun] Forcing full same-dataset LOO rerun.")
        if force_cross_dataset_loo:
            print("[rerun] Forcing full cross-dataset LOO rerun.")

    changed_total_known = isinstance(report.get("n_changed_total"), int)
    changed_total_positive = changed_total_known and int(report["n_changed_total"]) > 0

    if bool(sampling_cfg.get("enabled", False)) and bool(sampling_cfg.get("stratified", True)) and (
        force_same_dataset_loo or changed_total_positive
    ):
        raise ValueError(
            "Rerun mode does not support stratified document_sampling when gold labels change, "
            "because the sampled document subset itself may change. Disable stratified sampling or rerun from scratch."
        )

    _copy_results_tree(
        source_output_root=source_output_root,
        target_output_root=target_output_root,
        dry_run=dry_run,
    )

    zero_subdir = str(zero_cfg.get("output_subdir", "zero_shot"))
    loo_subdir = str(loo_cfg.get("output_subdir", "loo"))
    target_zero_output_dir = target_output_root / zero_subdir
    target_loo_output_dir = target_output_root / loo_subdir

    target_base_predictions = None
    if cli_base_predictions_path:
        target_base_predictions = Path(cli_base_predictions_path)
    elif target_zero_output_dir.joinpath("all_models.pkl").exists():
        target_base_predictions = target_zero_output_dir / "all_models.pkl"
    else:
        target_base_predictions = _rewrite_path_for_copied_root(
            str(paths_cfg.get("base_predictions_path", "")) if paths_cfg.get("base_predictions_path") else None,
            source_output_root=source_output_root,
            target_output_root=target_output_root,
        )

    target_paths_cfg = dict(paths_cfg)
    target_paths_cfg["output_root"] = str(target_output_root)
    if target_base_predictions is not None:
        target_paths_cfg["base_predictions_path"] = str(target_base_predictions)

    target_effective_cfg = dict(effective_cfg)
    target_effective_cfg["paths"] = target_paths_cfg
    target_effective_cfg["rerun"] = {
        **rerun_cfg,
        "enabled": True,
        "source_output_root": str(source_output_root),
        "target_output_root": str(target_output_root),
        "old_gold_labels": str(old_gold_path) if old_gold_path is not None else None,
        "new_gold_labels": str(gold_labels_path),
        "comparison_available": report["comparison_available"],
        "force_same_dataset_loo": force_same_dataset_loo,
        "force_cross_dataset_loo": force_cross_dataset_loo,
        "n_changed_total": report["n_changed_total"],
        "n_label_changed": report["n_label_changed"],
        "n_added": report["n_added"],
        "n_removed": report["n_removed"],
    }

    if not dry_run:
        snapshot_path = target_output_root / "config_snapshot.json"
        with snapshot_path.open("w", encoding="utf-8") as f:
            json.dump(target_effective_cfg, f, indent=2)
        print(f"[config] Wrote {snapshot_path}")
        _write_rerun_report(target_output_root, {**report, "changed_rows": changed_df.to_dict(orient="records")})
    else:
        snapshot_path = target_output_root / "config_snapshot.json"
        print(f"[rerun][dry-run] Would write {snapshot_path}")

    if (source_output_root / "regex_baseline").exists():
        _refresh_regex_baseline(
            config_snapshot_path=snapshot_path,
            output_dir=target_output_root / "regex_baseline",
            dry_run=dry_run,
        )

    valid_selections = {"random", "random_ra", "ping", "ping_ra", "cross_ping", "cross_ping_ra"}
    selection_scan_root = target_loo_output_dir if target_loo_output_dir.exists() else source_output_root / loo_subdir
    selection_dirs_present = [
        path.name
        for path in sorted(selection_scan_root.iterdir())
        if path.is_dir() and path.name in valid_selections
    ] if selection_scan_root.exists() else []
    same_dataset_selections = [
        selection for selection in selection_dirs_present if not _is_cross_ping_selection(selection)
    ]
    cross_dataset_selections = [
        selection for selection in selection_dirs_present if _is_cross_ping_selection(selection)
    ]
    rerun_selections: List[str] = []

    if (force_same_dataset_loo or changed_total_positive) and same_dataset_selections:
        rerun_selections.extend(same_dataset_selections)

    if force_cross_dataset_loo and cross_dataset_selections:
        rerun_selections.extend(cross_dataset_selections)

    if rerun_selections:
        rerun_reason_parts: List[str] = []
        if any(selection in same_dataset_selections for selection in rerun_selections):
            rerun_reason_parts.append("same-dataset selectors depend on updated target gold labels")
        if any(selection in cross_dataset_selections for selection in rerun_selections):
            rerun_reason_parts.append("cross-dataset selectors depend on updated source-dataset artifacts")
        rerun_reason = "; ".join(rerun_reason_parts)
        print(f"[rerun] Rerunning selections: {rerun_selections}")
        if rerun_reason:
            print(f"[rerun] Reason: {rerun_reason}")
        if dry_run:
            for selection in rerun_selections:
                print(f"[rerun][dry-run] Would invalidate {target_loo_output_dir / selection}")
        else:
            for selection in rerun_selections:
                _remove_path_if_exists(target_loo_output_dir / selection)

            rerun_loo_cfg = dict(loo_cfg)
            rerun_loo_cfg["run_selections"] = rerun_selections
            if target_base_predictions is None:
                raise FileNotFoundError(
                    "Unable to resolve base predictions for rerun mode. "
                    "Set paths.base_predictions_path or provide --base-predictions-path."
                )
            _run_loo(
                docs=docs,
                doc_ids=doc_ids,
                models=models,
                output_root=_ensure_dir(target_loo_output_dir),
                base_predictions_path=Path(target_base_predictions),
                gold_labels_path=gold_labels_path,
                paths_cfg=target_paths_cfg,
                zero_cfg=zero_cfg,
                loo_cfg=rerun_loo_cfg,
                dry_run=False,
                write_samples_txt=bool(artifacts_cfg.get("write_samples_txt", True)),
                sample_count_per_model=int(artifacts_cfg.get("sample_count_per_model", 1)),
                sample_random_state=int(artifacts_cfg.get("sample_random_state", 0)),
                write_error_log=bool(artifacts_cfg.get("write_error_log", True)),
            )

    if target_loo_output_dir.exists():
        eval_gold_df = _load_gold_labels_frame(gold_labels_path)
        if dry_run:
            print(f"[rerun][dry-run] Would rebuild LOO summaries under {target_loo_output_dir}")
        else:
            _rebuild_loo_summaries_from_existing_runs(
                loo_output_root=target_loo_output_dir,
                models=models,
                eval_gold_df=eval_gold_df,
            )

    print(f"[rerun] Finished. Updated results root: {target_output_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MIMIC zero-shot and LOO ICL experiments from config.")
    parser.add_argument("--config", required=True, help="Path to JSON config file.")
    parser.add_argument(
        "--stage",
        choices=["all", "zero-shot", "loo"],
        default="all",
        help="Which stage to run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse config and print planned runs without model inference.",
    )
    parser.add_argument(
        "--composition",
        default=None,
        help=(
            "Ping committee composition override. "
            "Supported: all, hi, lo, mix, hi3, lo3, mix3."
        ),
    )
    parser.add_argument(
        "--ping-ablation",
        default=None,
        help=(
            "Ping selector ablation override. "
            "Supported: default, a3_uncertainty_only_no_pong, "
            "a4_diversity_only_no_pong, a5_harmonic_no_pong, "
            "a6_granule_prior_round_robin_no_pong, "
            "a7_ping_only_reveal_no_compensation, "
            "a8_reveal_yhat_round_robin_no_pong, "
            "a9_reveal_yhat_round_robin_prob_update_no_pong."
        ),
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Override Config.paths.output_root for this run.",
    )
    parser.add_argument(
        "--base-predictions-path",
        default=None,
        help=(
            "Override path to base zero-shot predictions used by LOO. "
            "Useful when --output-root points to a fresh ablation directory."
        ),
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = _load_json(config_path)
    rerun_cfg = dict(cfg.get("rerun", {}))
    rerun_enabled = bool(rerun_cfg.get("enabled", False))

    models = list(cfg.get("models", []))
    if not models:
        raise ValueError("Config must include non-empty 'models'.")

    paths_cfg = dict(cfg.get("paths", {}))
    data_dir, gold_labels_path, paths_cfg = _resolve_dataset_paths(
        paths_cfg,
        base_dir=config_path.parent.resolve(),
    )

    configured_output_root = Path(paths_cfg.get("output_root", "results/mimic_pipeline"))
    if rerun_enabled:
        output_root = configured_output_root
        paths_cfg["output_root"] = str(output_root)
    else:
        output_root_value = args.output_root if args.output_root else paths_cfg.get("output_root", "results/mimic_pipeline")
        output_root = _ensure_dir(Path(output_root_value))
        paths_cfg["output_root"] = str(output_root)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    zero_cfg = dict(cfg.get("zero_shot", {}))
    loo_cfg = dict(cfg.get("loo", {}))
    artifacts_cfg = dict(cfg.get("artifacts", {}))
    sampling_cfg = dict(cfg.get("document_sampling", {}))

    if "prompt_template" not in zero_cfg:
        zero_cfg["prompt_template"] = DEFAULT_ZERO_SHOT_PROMPT
    zero_cfg["prompt_template"] = _resolve_zero_shot_prompt_template(cfg, zero_cfg)

    if "query_template" not in loo_cfg:
        loo_cfg["query_template"] = DEFAULT_QUERY_TEMPLATE

    if "prompt_kwargs" not in loo_cfg:
        loo_cfg["prompt_kwargs"] = {}
    if "instruction" not in loo_cfg["prompt_kwargs"]:
        loo_cfg["prompt_kwargs"]["instruction"] = DEFAULT_ICL_INSTRUCTION
    loo_cfg["prompt_kwargs"]["instruction"] = _resolve_icl_instruction(cfg, loo_cfg)
    if "include_group_headers" not in loo_cfg["prompt_kwargs"]:
        loo_cfg["prompt_kwargs"]["include_group_headers"] = False
    if "ping_composition" not in loo_cfg:
        loo_cfg["ping_composition"] = DEFAULT_PING_COMPOSITION
    if "ping_committee_size" not in loo_cfg:
        loo_cfg["ping_committee_size"] = DEFAULT_PING_COMMITTEE_SIZE
    if "ping_ablation" not in loo_cfg:
        loo_cfg["ping_ablation"] = "default"
    if args.composition is not None:
        loo_cfg["ping_composition"] = args.composition
        # Composition is defined for ping-family selections only.
        ping_like_selections = [
            selection
            for selection in loo_cfg.get("run_selections", [])
            if _is_ping_selection(selection)
        ]
        loo_cfg["run_selections"] = ping_like_selections or ["ping"]

    ablation_raw = args.ping_ablation if args.ping_ablation is not None else loo_cfg.get("ping_ablation")
    applied_ablation = _apply_ping_ablation(loo_cfg, ablation_raw)
    if applied_ablation and applied_ablation != "default":
        print(f"[loo] Ping ablation: {applied_ablation}")
    if args.ping_ablation is not None and applied_ablation and applied_ablation != "default":
        # Explicit CLI ablation runs are intended for ping-family experiments.
        ping_like_selections = [
            selection
            for selection in loo_cfg.get("run_selections", [])
            if _is_ping_selection(selection)
        ]
        loo_cfg["run_selections"] = ping_like_selections or ["ping"]

    loo_cfg["embed_device"] = _resolve_device(loo_cfg.get("embed_device", "auto"))
    loo_cfg["ping_committee_size"] = int(loo_cfg.get("ping_committee_size", DEFAULT_PING_COMMITTEE_SIZE))

    write_samples_txt = bool(artifacts_cfg.get("write_samples_txt", True))
    sample_count_per_model = int(artifacts_cfg.get("sample_count_per_model", 1))
    sample_random_state = int(artifacts_cfg.get("sample_random_state", 0))
    write_error_log = bool(artifacts_cfg.get("write_error_log", True))
    sampling_enabled = bool(sampling_cfg.get("enabled", False))
    sampling_fraction = float(sampling_cfg.get("fraction", 1.0))
    sampling_stratified = bool(sampling_cfg.get("stratified", True))
    sampling_random_state = int(sampling_cfg.get("random_state", 42))

    if sample_count_per_model < 0:
        raise ValueError("artifacts.sample_count_per_model must be >= 0.")
    if sampling_fraction <= 0.0 or sampling_fraction > 1.0:
        raise ValueError("document_sampling.fraction must be in (0, 1].")

    artifacts_cfg["write_samples_txt"] = write_samples_txt
    artifacts_cfg["sample_count_per_model"] = sample_count_per_model
    artifacts_cfg["sample_random_state"] = sample_random_state
    artifacts_cfg["write_error_log"] = write_error_log
    sampling_cfg["enabled"] = sampling_enabled
    sampling_cfg["fraction"] = sampling_fraction
    sampling_cfg["stratified"] = sampling_stratified
    sampling_cfg["random_state"] = sampling_random_state

    effective_cfg = dict(cfg)
    effective_cfg["paths"] = paths_cfg
    effective_cfg["zero_shot"] = zero_cfg
    effective_cfg["loo"] = loo_cfg
    effective_cfg["artifacts"] = artifacts_cfg
    effective_cfg["document_sampling"] = sampling_cfg
    effective_cfg["rerun"] = rerun_cfg

    # Persist the effective config used for this run.
    if not args.dry_run and not rerun_enabled:
        snapshot_path = output_root / "config_snapshot.json"
        with snapshot_path.open("w", encoding="utf-8") as f:
            json.dump(effective_cfg, f, indent=2)
        print(f"[config] Wrote {snapshot_path}")

    all_docs = load_documents(str(data_dir))
    if not all_docs:
        raise RuntimeError(f"No documents found in {data_dir}")
    docs = _sample_documents(
        all_docs,
        sampling_cfg=sampling_cfg,
        gold_labels_path=gold_labels_path,
    )
    doc_ids = [str(d["file"]) for d in docs]
    print(f"[dataset] Selected dataset: {paths_cfg.get('dataset', 'mimic')}")
    print(f"Loaded {len(all_docs)} documents from {data_dir}")
    if len(docs) != len(all_docs):
        print(f"[sampling] Active subset size: {len(docs)} documents.")

    if rerun_enabled:
        if gold_labels_path is None:
            raise ValueError("Config.paths.gold_labels is required when rerun.enabled=true.")
        _run_gold_label_rerun(
            config_path=config_path,
            effective_cfg=effective_cfg,
            rerun_cfg=rerun_cfg,
            paths_cfg=paths_cfg,
            docs=docs,
            doc_ids=doc_ids,
            models=models,
            gold_labels_path=gold_labels_path,
            zero_cfg=zero_cfg,
            loo_cfg=loo_cfg,
            artifacts_cfg=artifacts_cfg,
            sampling_cfg=sampling_cfg,
            cli_output_root=args.output_root,
            cli_base_predictions_path=args.base_predictions_path,
            dry_run=args.dry_run,
        )
        print("Done.")
        return

    run_zero_stage = args.stage in {"all", "zero-shot"} and bool(zero_cfg.get("enabled", True))
    run_loo_stage = args.stage in {"all", "loo"} and bool(loo_cfg.get("enabled", True))

    zero_subdir = zero_cfg.get("output_subdir", "zero_shot")
    loo_subdir = loo_cfg.get("output_subdir", "loo")
    zero_output_dir = output_root / zero_subdir
    loo_output_dir = output_root / loo_subdir

    if run_zero_stage:
        _run_zero_shot(
            docs=docs,
            doc_ids=doc_ids,
            models=models,
            output_dir=zero_output_dir,
            prompt_template=zero_cfg.get("prompt_template", DEFAULT_ZERO_SHOT_PROMPT),
            dry_run=args.dry_run,
            write_samples_txt=write_samples_txt,
            sample_count_per_model=sample_count_per_model,
            sample_random_state=sample_random_state,
            write_error_log=write_error_log,
        )

    base_predictions_path_cfg = args.base_predictions_path if args.base_predictions_path else paths_cfg.get("base_predictions_path")
    if base_predictions_path_cfg:
        base_predictions_path = Path(base_predictions_path_cfg)
    else:
        base_predictions_path = zero_output_dir / "all_models.pkl"
        if run_loo_stage and not run_zero_stage and args.output_root is not None:
            fallback_base_predictions = configured_output_root / zero_subdir / "all_models.pkl"
            if fallback_base_predictions.exists():
                base_predictions_path = fallback_base_predictions
                print(f"[loo] Using base predictions from configured root: {base_predictions_path}")
    paths_cfg["base_predictions_path"] = str(base_predictions_path)

    if run_loo_stage:
        if gold_labels_path is None:
            raise ValueError("Config.paths.gold_labels is required when running LOO stage.")
        _run_loo(
            docs=docs,
            doc_ids=doc_ids,
            models=models,
            output_root=_ensure_dir(loo_output_dir),
            base_predictions_path=base_predictions_path,
            gold_labels_path=gold_labels_path,
            paths_cfg=paths_cfg,
            zero_cfg=zero_cfg,
            loo_cfg=loo_cfg,
            dry_run=args.dry_run,
            write_samples_txt=write_samples_txt,
            sample_count_per_model=sample_count_per_model,
            sample_random_state=sample_random_state,
            write_error_log=write_error_log,
        )

    print("Done.")


if __name__ == "__main__":
    main()
