"""Build consolidated LOO evaluation tables for downstream analysis notebooks."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from .evaluation_utils import (
    attach_gold_and_preds,
    collapse_predictions_by_document,
    majority_vote_with_seed_priority,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MACRO_LABELS = ["early", "late", "unrelated"]
LABEL_TO_INT = {label: idx for idx, label in enumerate(MACRO_LABELS)}
UNKNOWN_PRED_CODE = len(MACRO_LABELS)

PRIMARY_M = 2
EXPECTED_SEEDS = [0, 30, 500, 6370, 17893]
SENSITIVITY_M_VALUES = [2, 4, 6, 8, 10]

COMMITTEE_MODEL_NAME = "COMMITTEE"
REGEX_MODEL_NAME = "regex_baseline"

PROMPT_REGIME_ZERO_SHOT = "zero-shot"
PROMPT_REGIME_LABEL_ONLY = "label-only ICL"
PROMPT_REGIME_RATIONALE = "rationale-augmented ICL"

SEED_K_PREFIX = "seed"

DEFAULT_VARIANT_ROOTS: Sequence[Tuple[str, str, str]] = (
    ("MIMIC", "short", "results/mimic_streamlined_pipeline_small_models_upd_short"),
    ("MIMIC", "long", "results/mimic_streamlined_pipeline_small_models_upd_long"),
    ("Indian", "short", "results/mimic_streamlined_pipeline_small_models_indic_upd_short"),
    ("Indian", "long", "results/mimic_streamlined_pipeline_small_models_indic_upd_long"),
)


def _stable_seed(*parts: object, base: int = 0) -> int:
    text = "||".join("" if part is None else str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(base) + (int(digest[:12], 16) % 100000)


@dataclass(frozen=True)
class VariantSpec:
    cohort: str
    prompt_variant: str
    output_root: Path
    config_snapshot_path: Path
    gold_path: Path
    models: Tuple[str, ...]

    @property
    def zero_shot_dir(self) -> Path:
        return self.output_root / "zero_shot"

    @property
    def loo_dir(self) -> Path:
        return self.output_root / "loo"

    @property
    def regex_dir(self) -> Path:
        return self.output_root / "regex_baseline"


@dataclass(frozen=True)
class ConditionSpec:
    key: str
    source_key: str
    regime: Optional[str]
    prompt_regime: str
    expect_five_seeds: bool
    zero_shot: bool = False


CONDITION_SPECS: Sequence[ConditionSpec] = (
    ConditionSpec(
        key="zero_shot",
        source_key="zero_shot",
        regime=None,
        prompt_regime=PROMPT_REGIME_ZERO_SHOT,
        expect_five_seeds=False,
        zero_shot=True,
    ),
    ConditionSpec(
        key="label_only_with_update",
        source_key="ping",
        regime="WithUpdate",
        prompt_regime=PROMPT_REGIME_LABEL_ONLY,
        expect_five_seeds=True,
    ),
    ConditionSpec(
        key="rationale_with_update",
        source_key="ping_ra",
        regime="WithUpdate",
        prompt_regime=PROMPT_REGIME_RATIONALE,
        expect_five_seeds=True,
    ),
    ConditionSpec(
        key="label_only_random",
        source_key="random",
        regime="Random",
        prompt_regime=PROMPT_REGIME_LABEL_ONLY,
        expect_five_seeds=True,
    ),
    ConditionSpec(
        key="rationale_random",
        source_key="random_ra",
        regime="Random",
        prompt_regime=PROMPT_REGIME_RATIONALE,
        expect_five_seeds=True,
    ),
)


def _read_json(path: Path) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_gold_labels(gold_path: Path) -> pd.DataFrame:
    path = Path(gold_path)
    suffix = path.suffix.lower()

    if suffix in {".pkl", ".pickle"}:
        frame = pd.read_pickle(path)
    else:
        try:
            frame = pd.read_csv(path)
        except UnicodeDecodeError:
            frame = pd.read_pickle(path)

    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"Gold labels did not load as a DataFrame: {path}")
    return frame.copy()


def discover_variant_specs(
    variant_roots: Sequence[Tuple[str, str, str]] = DEFAULT_VARIANT_ROOTS,
) -> List[VariantSpec]:
    specs: List[VariantSpec] = []
    for cohort, prompt_variant, rel_root in variant_roots:
        output_root = PROJECT_ROOT / rel_root
        config_snapshot_path = output_root / "config_snapshot.json"
        if not output_root.exists():
            raise FileNotFoundError(f"Missing output root: {output_root}")
        if not config_snapshot_path.exists():
            raise FileNotFoundError(f"Missing config snapshot: {config_snapshot_path}")

        cfg = _read_json(config_snapshot_path)
        paths_cfg = cfg.get("paths", {})
        gold_path = PROJECT_ROOT / str(paths_cfg.get("gold_labels"))
        models = tuple(str(model) for model in cfg.get("models", []))
        if not gold_path.exists():
            raise FileNotFoundError(f"Missing gold labels for {output_root}: {gold_path}")
        if not models:
            raise ValueError(f"No models listed in {config_snapshot_path}")

        specs.append(
            VariantSpec(
                cohort=cohort,
                prompt_variant=prompt_variant,
                output_root=output_root,
                config_snapshot_path=config_snapshot_path,
                gold_path=gold_path,
                models=models,
            )
        )
    return specs


def _parse_seed_m(dirname: str) -> Tuple[Optional[int], Optional[int]]:
    text = str(dirname).strip()
    if not text.startswith(SEED_K_PREFIX) or "_k" not in text:
        return None, None
    seed_text, m_text = text[len(SEED_K_PREFIX) :].split("_k", 1)
    try:
        return int(seed_text), int(m_text)
    except ValueError:
        return None, None


def _ensure_base_columns(df: pd.DataFrame) -> pd.DataFrame:
    required = ["file", "model", "raw_output", "prompt"]
    work = df.copy()
    for column in required:
        if column not in work.columns:
            work[column] = pd.NA
    return work[required]


def _read_pickle(path: Path) -> pd.DataFrame:
    try:
        return pd.read_pickle(path)
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Failed reading pickle {path}: {exc}") from exc


def _load_prediction_pickles(run_dir: Path) -> pd.DataFrame:
    if not run_dir.exists():
        return pd.DataFrame(columns=["file", "model", "raw_output", "prompt"])

    result_files = sorted(run_dir.glob("results_*.pkl"))
    if result_files:
        frames = [_ensure_base_columns(_read_pickle(path)) for path in result_files]
        if not frames:
            return pd.DataFrame(columns=["file", "model", "raw_output", "prompt"])
        return pd.concat(frames, ignore_index=True)

    all_models_path = run_dir / "all_models.pkl"
    if all_models_path.exists():
        return _ensure_base_columns(_read_pickle(all_models_path))

    return pd.DataFrame(columns=["file", "model", "raw_output", "prompt"])


def load_zero_shot_predictions(spec: VariantSpec) -> pd.DataFrame:
    frame = _load_prediction_pickles(spec.zero_shot_dir)
    if frame.empty:
        return frame.assign(seed=pd.Series(dtype="Int64"), m=pd.Series(dtype="Int64"))
    frame = frame[frame["model"].isin(spec.models)].copy()
    frame["seed"] = pd.NA
    frame["m"] = pd.NA
    return frame


def load_icl_predictions(spec: VariantSpec, source_key: str) -> pd.DataFrame:
    method_dir = spec.loo_dir / source_key
    if not method_dir.exists():
        return pd.DataFrame(columns=["file", "model", "raw_output", "prompt", "seed", "m"])

    frames: List[pd.DataFrame] = []
    for run_dir in sorted(path for path in method_dir.iterdir() if path.is_dir()):
        seed, m = _parse_seed_m(run_dir.name)
        if seed is None or m is None:
            continue
        frame = _load_prediction_pickles(run_dir)
        if frame.empty:
            continue
        frame = frame[frame["model"].isin(spec.models)].copy()
        frame["seed"] = int(seed)
        frame["m"] = int(m)
        frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=["file", "model", "raw_output", "prompt", "seed", "m"])
    return pd.concat(frames, ignore_index=True)


def load_condition_source_frames(spec: VariantSpec) -> Dict[str, pd.DataFrame]:
    return {
        "zero_shot": load_zero_shot_predictions(spec),
        "ping": load_icl_predictions(spec, "ping"),
        "ping_ra": load_icl_predictions(spec, "ping_ra"),
        "random": load_icl_predictions(spec, "random"),
        "random_ra": load_icl_predictions(spec, "random_ra"),
    }


def merge_with_gold(predictions: pd.DataFrame, gold_df: pd.DataFrame) -> pd.DataFrame:
    if predictions is None or predictions.empty:
        return pd.DataFrame(
            columns=["file", "file_pred", "model", "raw_output", "prompt", "pred", "label", "seed", "m"]
        )

    merged = attach_gold_and_preds(
        predictions,
        gold_df,
        pred_col="raw_output",
        file_col="file",
        on="file",
        case_insensitive=True,
        prefer_longest_match=True,
        join_how="inner",
    )
    expected = ["file", "file_pred", "model", "raw_output", "prompt", "pred", "label", "seed", "m"]
    for column in expected:
        if column not in merged.columns:
            merged[column] = pd.NA
    merged["file"] = merged["file"].astype(str)
    merged["model"] = merged["model"].astype(str)
    if "seed" in merged.columns:
        merged["seed"] = pd.to_numeric(merged["seed"], errors="coerce").astype("Int64")
    if "m" in merged.columns:
        merged["m"] = pd.to_numeric(merged["m"], errors="coerce").astype("Int64")
    return merged[expected]


def _collapse_model_predictions(merged: pd.DataFrame) -> pd.DataFrame:
    if merged is None or merged.empty:
        return pd.DataFrame(columns=["model", "file", "label", "pred"])

    keep_first_cols = [column for column in ["prompt"] if column in merged.columns]
    return collapse_predictions_by_document(
        merged,
        doc_col="file",
        pred_col="pred",
        label_col="label",
        seed_col="seed",
        group_cols=["model"],
        keep_first_cols=keep_first_cols,
    )


def _build_seed_level_committee(merged: pd.DataFrame) -> pd.DataFrame:
    if merged is None or merged.empty:
        return pd.DataFrame(columns=["model", "file", "label", "pred", "seed"])

    work = merged.dropna(subset=["file", "model", "pred", "label"]).copy()
    if work.empty:
        return pd.DataFrame(columns=["model", "file", "label", "pred", "seed"])

    if "seed" not in work.columns or work["seed"].isna().all():
        rows = []
        for doc, group in work.groupby("file", sort=True):
            vote = majority_vote_with_seed_priority(group["pred"])
            rows.append(
                {
                    "model": COMMITTEE_MODEL_NAME,
                    "file": doc,
                    "label": group["label"].iloc[0],
                    "pred": vote,
                    "seed": pd.NA,
                }
            )
        return pd.DataFrame(rows)

    rows = []
    for (seed, doc), group in work.groupby(["seed", "file"], dropna=False, sort=True):
        vote = majority_vote_with_seed_priority(group["pred"])
        rows.append(
            {
                "model": COMMITTEE_MODEL_NAME,
                "file": doc,
                "label": group["label"].iloc[0],
                "pred": vote,
                "seed": seed,
            }
        )
    return pd.DataFrame(rows)


def _collapse_committee_predictions(seed_level_committee: pd.DataFrame) -> pd.DataFrame:
    if seed_level_committee is None or seed_level_committee.empty:
        return pd.DataFrame(columns=["model", "file", "label", "pred"])

    if "seed" not in seed_level_committee.columns or seed_level_committee["seed"].isna().all():
        return seed_level_committee[["model", "file", "label", "pred"]].drop_duplicates(subset=["model", "file"])

    return collapse_predictions_by_document(
        seed_level_committee,
        doc_col="file",
        pred_col="pred",
        label_col="label",
        seed_col="seed",
        group_cols=["model"],
    )


def _encode_predictions(
    y_true: Sequence[object],
    y_pred: Sequence[object],
    labels: Sequence[str] = MACRO_LABELS,
) -> Tuple[np.ndarray, np.ndarray]:
    label_to_int = {str(label): idx for idx, label in enumerate(labels)}
    true_codes = np.array([label_to_int.get(str(value), -1) for value in y_true], dtype=np.int16)
    pred_codes = np.array([label_to_int.get(str(value), UNKNOWN_PRED_CODE) for value in y_pred], dtype=np.int16)
    mask = true_codes >= 0
    return true_codes[mask], pred_codes[mask]


def _macro_f1_from_encoded_arrays(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_labels: int = len(MACRO_LABELS),
) -> float:
    if y_true.size == 0:
        return np.nan

    counts = np.zeros((n_labels, n_labels + 1), dtype=np.int64)
    np.add.at(counts, (y_true, y_pred), 1)
    tp = counts[np.arange(n_labels), np.arange(n_labels)].astype(float)
    pred_pos = counts[:, :n_labels].sum(axis=0).astype(float)
    true_pos = counts.sum(axis=1).astype(float)
    fp = pred_pos - tp
    fn = true_pos - tp

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) > 0,
    )
    return float(f1.mean())


def _macro_f1_from_sample_arrays(
    sampled_true: np.ndarray,
    sampled_pred: np.ndarray,
    n_labels: int = len(MACRO_LABELS),
) -> np.ndarray:
    take = sampled_true.shape[0]
    counts = np.zeros((take, n_labels, n_labels + 1), dtype=np.int32)
    for true_code in range(n_labels):
        true_mask = sampled_true == true_code
        for pred_code in range(n_labels + 1):
            counts[:, true_code, pred_code] = np.sum(true_mask & (sampled_pred == pred_code), axis=1)

    tp = counts[:, np.arange(n_labels), np.arange(n_labels)].astype(float)
    pred_pos = counts[:, :, :n_labels].sum(axis=1).astype(float)
    true_pos = counts.sum(axis=2).astype(float)
    fp = pred_pos - tp
    fn = true_pos - tp

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) > 0,
    )
    return f1.mean(axis=1)


def bootstrap_macro_f1_ci(
    predictions: pd.DataFrame,
    *,
    B: int,
    alpha: float,
    random_state: int,
) -> Tuple[float, float, float, int]:
    if predictions is None or predictions.empty:
        return np.nan, np.nan, np.nan, 0

    y_true, y_pred = _encode_predictions(predictions["label"], predictions["pred"])
    n_docs = int(len(y_true))
    if n_docs == 0:
        return np.nan, np.nan, np.nan, 0

    observed = _macro_f1_from_encoded_arrays(y_true, y_pred)
    if B <= 0:
        return observed, observed, observed, n_docs

    rng = np.random.default_rng(random_state)
    chunk_size = 256
    draws: List[np.ndarray] = []
    remaining = int(B)
    while remaining > 0:
        take = min(chunk_size, remaining)
        indices = rng.integers(0, n_docs, size=(take, n_docs))
        sample_true = y_true[indices]
        sample_pred = y_pred[indices]
        draws.append(_macro_f1_from_sample_arrays(sample_true, sample_pred))
        remaining -= take

    boot = np.concatenate(draws)
    ci_low = float(np.quantile(boot, alpha / 2.0))
    ci_high = float(np.quantile(boot, 1.0 - (alpha / 2.0)))
    return observed, ci_low, ci_high, n_docs


def paired_bootstrap_comparison(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    B: int,
    alpha: float,
    random_state: int,
) -> Dict[str, float]:
    columns = ["delta_f1", "bootstrap_ci_low", "bootstrap_ci_high", "bootstrap_p", "n_docs"]
    empty = {column: np.nan for column in columns}
    empty["bootstrap_p"] = np.nan
    empty["n_docs"] = 0

    if left is None or left.empty or right is None or right.empty:
        return empty

    lhs = left[["file", "label", "pred"]].drop_duplicates(subset=["file"]).set_index("file")
    rhs = right[["file", "label", "pred"]].drop_duplicates(subset=["file"]).set_index("file")
    common = lhs.index.intersection(rhs.index)
    if common.empty:
        return empty

    lhs = lhs.loc[common].reset_index()
    rhs = rhs.loc[common].reset_index()

    y_true_left, y_pred_left = _encode_predictions(lhs["label"], lhs["pred"])
    y_true_right, y_pred_right = _encode_predictions(rhs["label"], rhs["pred"])
    if len(y_true_left) != len(y_true_right):
        return empty
    if len(y_true_left) == 0:
        return empty

    n_docs = int(len(y_true_left))
    obs_left = _macro_f1_from_encoded_arrays(y_true_left, y_pred_left)
    obs_right = _macro_f1_from_encoded_arrays(y_true_right, y_pred_right)
    observed_delta = float(obs_left - obs_right)

    if B <= 0:
        return {
            "delta_f1": observed_delta,
            "bootstrap_ci_low": observed_delta,
            "bootstrap_ci_high": observed_delta,
            "bootstrap_p": 1.0,
            "n_docs": n_docs,
        }

    rng = np.random.default_rng(random_state)
    chunk_size = 256
    draws: List[np.ndarray] = []
    remaining = int(B)
    while remaining > 0:
        take = min(chunk_size, remaining)
        indices = rng.integers(0, n_docs, size=(take, n_docs))
        sample_true = y_true_left[indices]
        sample_pred_left = y_pred_left[indices]
        sample_pred_right = y_pred_right[indices]
        left_scores = _macro_f1_from_sample_arrays(sample_true, sample_pred_left)
        right_scores = _macro_f1_from_sample_arrays(sample_true, sample_pred_right)
        draws.append(left_scores - right_scores)
        remaining -= take

    boot = np.concatenate(draws)
    ci_low = float(np.quantile(boot, alpha / 2.0))
    ci_high = float(np.quantile(boot, 1.0 - (alpha / 2.0)))
    p_lower = (np.sum(boot <= 0.0) + 1.0) / (len(boot) + 1.0)
    p_upper = (np.sum(boot >= 0.0) + 1.0) / (len(boot) + 1.0)
    bootstrap_p = float(min(1.0, 2.0 * min(p_lower, p_upper)))
    return {
        "delta_f1": observed_delta,
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "bootstrap_p": bootstrap_p,
        "n_docs": n_docs,
    }


def _seed_coverage(merged: pd.DataFrame, primary_m: int) -> Dict[int, List[int]]:
    if merged is None or merged.empty or "m" not in merged.columns or "seed" not in merged.columns:
        return {}
    work = merged.dropna(subset=["m", "seed"]).copy()
    work = work[work["m"].astype(int) == int(primary_m)]
    coverage: Dict[int, List[int]] = {}
    for model, group in work.groupby("model", sort=True):
        coverage_key = str(model)
        coverage[coverage_key] = sorted(group["seed"].dropna().astype(int).unique().tolist())
    return coverage


def _has_complete_seed_coverage(merged: pd.DataFrame, primary_m: int, expected_seeds: Sequence[int]) -> bool:
    coverage = _seed_coverage(merged, primary_m)
    if not coverage:
        return False
    expected = sorted(int(seed) for seed in expected_seeds)
    return all(seeds == expected for seeds in coverage.values())


def _filter_primary_condition(
    merged: pd.DataFrame,
    condition: ConditionSpec,
    *,
    primary_m: int,
    expected_seeds: Sequence[int],
) -> Tuple[pd.DataFrame, bool]:
    if merged is None or merged.empty:
        return pd.DataFrame(columns=["file", "model", "label", "pred", "seed", "m"]), False

    work = merged.copy()
    if condition.zero_shot:
        return work, True

    work = work.dropna(subset=["m"]).copy()
    if work.empty:
        return work, False
    work = work[work["m"].astype(int) == int(primary_m)].copy()
    if work.empty:
        return work, False

    is_complete = _has_complete_seed_coverage(work, primary_m, expected_seeds) if condition.expect_five_seeds else True
    return work, is_complete


def ensure_regex_artifacts(
    specs: Sequence[VariantSpec],
    *,
    force_recalc: bool,
    warnings: List[str],
) -> None:
    script_path = PROJECT_ROOT / "evaluate_mimic_regex_baseline.py"
    for spec in specs:
        summary_path = spec.regex_dir / "regex_baseline_summary.json"
        if summary_path.exists() and not force_recalc:
            continue
        spec.regex_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(script_path),
            "--config",
            str(spec.config_snapshot_path),
            "--output-dir",
            str(spec.regex_dir),
        ]
        try:
            subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:  # pragma: no cover - depends on local env
            warnings.append(
                "Regex baseline recomputation failed for "
                f"{spec.output_root}: {exc.stderr.strip() or exc.stdout.strip() or exc}"
            )


def load_regex_predictions(spec: VariantSpec) -> pd.DataFrame:
    predictions_path = spec.regex_dir / "regex_baseline_predictions.csv"
    if not predictions_path.exists():
        return pd.DataFrame(columns=["file", "label", "pred", "model"])
    frame = pd.read_csv(predictions_path)
    if frame.empty:
        return pd.DataFrame(columns=["file", "label", "pred", "model"])
    frame = frame.rename(columns={"gold_short": "label", "pred_short": "pred"})
    frame["model"] = REGEX_MODEL_NAME
    return frame[["file", "label", "pred", "model"]].copy()


def _regex_row_for_cohort(
    cohort: str,
    regex_predictions: pd.DataFrame,
    *,
    bootstrap_reps: int,
    alpha: float,
    random_state: int,
) -> pd.DataFrame:
    if regex_predictions is None or regex_predictions.empty:
        return pd.DataFrame(
            [
                {
                    "cohort": cohort,
                    "prompt_variant": pd.NA,
                    "model": REGEX_MODEL_NAME,
                    "regime": pd.NA,
                    "prompt_regime": pd.NA,
                    "macro_f1": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "n_docs": 0,
                }
            ]
        )

    observed, ci_low, ci_high, n_docs = bootstrap_macro_f1_ci(
        regex_predictions,
        B=bootstrap_reps,
        alpha=alpha,
        random_state=random_state,
    )
    return pd.DataFrame(
        [
            {
                "cohort": cohort,
                "prompt_variant": pd.NA,
                "model": REGEX_MODEL_NAME,
                "regime": pd.NA,
                "prompt_regime": pd.NA,
                "macro_f1": observed,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n_docs": n_docs,
            }
        ]
    )


def _empty_table_1_rows(spec: VariantSpec, condition: ConditionSpec) -> pd.DataFrame:
    models = list(spec.models) + [COMMITTEE_MODEL_NAME]
    rows = []
    for model_name in models:
        rows.append(
            {
                "cohort": spec.cohort,
                "prompt_variant": spec.prompt_variant,
                "model": model_name,
                "regime": condition.regime,
                "prompt_regime": condition.prompt_regime,
                "macro_f1": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "n_docs": 0,
            }
        )
    return pd.DataFrame(rows)


def _compute_table_1_rows(
    spec: VariantSpec,
    condition: ConditionSpec,
    merged_primary: pd.DataFrame,
    *,
    bootstrap_reps: int,
    alpha: float,
    random_state: int,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    if merged_primary is None or merged_primary.empty:
        return _empty_table_1_rows(spec, condition), {}

    model_predictions = _collapse_model_predictions(merged_primary)
    seed_level_committee = _build_seed_level_committee(merged_primary)
    collapsed_committee = _collapse_committee_predictions(seed_level_committee)

    frames_by_model: Dict[str, pd.DataFrame] = {}
    rows = []

    for model_name in spec.models:
        model_frame = model_predictions[model_predictions["model"] == model_name].copy()
        frames_by_model[str(model_name)] = model_frame
        observed, ci_low, ci_high, n_docs = bootstrap_macro_f1_ci(
            model_frame,
            B=bootstrap_reps,
            alpha=alpha,
            random_state=_stable_seed(
                spec.cohort,
                spec.prompt_variant,
                condition.key,
                model_name,
                base=random_state,
            ),
        )
        rows.append(
            {
                "cohort": spec.cohort,
                "prompt_variant": spec.prompt_variant,
                "model": model_name,
                "regime": condition.regime,
                "prompt_regime": condition.prompt_regime,
                "macro_f1": observed,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n_docs": n_docs,
            }
        )

    frames_by_model[COMMITTEE_MODEL_NAME] = collapsed_committee
    observed, ci_low, ci_high, n_docs = bootstrap_macro_f1_ci(
        collapsed_committee,
        B=bootstrap_reps,
        alpha=alpha,
        random_state=_stable_seed(
            spec.cohort,
            spec.prompt_variant,
            condition.key,
            COMMITTEE_MODEL_NAME,
            base=random_state,
        ),
    )
    rows.append(
        {
            "cohort": spec.cohort,
            "prompt_variant": spec.prompt_variant,
            "model": COMMITTEE_MODEL_NAME,
            "regime": condition.regime,
            "prompt_regime": condition.prompt_regime,
            "macro_f1": observed,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "n_docs": n_docs,
        }
    )
    return pd.DataFrame(rows), frames_by_model


def _compute_table_4_rows(
    spec: VariantSpec,
    condition: ConditionSpec,
    merged_runs: pd.DataFrame,
) -> pd.DataFrame:
    columns = ["cohort", "prompt_variant", "model", "regime", "prompt_regime", "m", "seed", "macro_f1"]
    if merged_runs is None or merged_runs.empty or condition.zero_shot:
        return pd.DataFrame(columns=columns)

    work = merged_runs.dropna(subset=["seed", "m"]).copy()
    if work.empty:
        return pd.DataFrame(columns=columns)
    work["seed"] = work["seed"].astype(int)
    work["m"] = work["m"].astype(int)
    work = work[work["m"].isin(SENSITIVITY_M_VALUES)].copy()
    if work.empty:
        return pd.DataFrame(columns=columns)

    rows: List[Dict[str, object]] = []
    for (seed, m), group in work.groupby(["seed", "m"], sort=True):
        # Each saved run already has one prediction per model/document; only guard
        # against accidental duplicate rows from partial reruns.
        per_model = group.drop_duplicates(subset=["model", "file"], keep="first")
        for model_name, model_group in per_model.groupby("model", sort=True):
            score = _macro_f1_from_encoded_arrays(*_encode_predictions(model_group["label"], model_group["pred"]))
            rows.append(
                {
                    "cohort": spec.cohort,
                    "prompt_variant": spec.prompt_variant,
                    "model": model_name,
                    "regime": condition.regime,
                    "prompt_regime": condition.prompt_regime,
                    "m": int(m),
                    "seed": int(seed),
                    "macro_f1": score,
                }
            )

        committee_seed = _build_seed_level_committee(group)
        if not committee_seed.empty:
            committee_score = _macro_f1_from_encoded_arrays(
                *_encode_predictions(committee_seed["label"], committee_seed["pred"])
            )
            rows.append(
                {
                    "cohort": spec.cohort,
                    "prompt_variant": spec.prompt_variant,
                    "model": COMMITTEE_MODEL_NAME,
                    "regime": condition.regime,
                    "prompt_regime": condition.prompt_regime,
                    "m": int(m),
                    "seed": int(seed),
                    "macro_f1": committee_score,
                }
            )

    return pd.DataFrame(rows, columns=columns)


def _best_condition_for_model(
    table_1: pd.DataFrame,
    *,
    cohort: str,
    prompt_variant: str,
    model_name: str,
) -> Optional[Tuple[str, str]]:
    subset = table_1[
        (table_1["cohort"] == cohort)
        & (table_1["prompt_variant"] == prompt_variant)
        & (table_1["model"] == model_name)
        & table_1["prompt_regime"].isin([PROMPT_REGIME_LABEL_ONLY, PROMPT_REGIME_RATIONALE])
    ].copy()
    subset = subset.dropna(subset=["macro_f1"])
    if subset.empty:
        return None
    best = subset.sort_values(["macro_f1", "prompt_regime", "regime"], ascending=[False, True, True]).iloc[0]
    return str(best["regime"]), str(best["prompt_regime"])


def _build_table_2_variant(
    table_1: pd.DataFrame,
    prediction_store: Mapping[Tuple[str, str, str, str], pd.DataFrame],
    *,
    include_committee: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    table_2_rows: List[Dict[str, object]] = []
    best_rows: List[Dict[str, object]] = []

    for cohort in sorted(table_1["cohort"].dropna().unique().tolist()):
        cohort_subset = table_1[
            (table_1["cohort"] == cohort)
            & (table_1["model"] != REGEX_MODEL_NAME)
        ].dropna(subset=["macro_f1"])
        if not include_committee:
            cohort_subset = cohort_subset[cohort_subset["model"] != COMMITTEE_MODEL_NAME]
        if cohort_subset.empty:
            continue

        best = cohort_subset.sort_values(["macro_f1", "prompt_variant", "model"], ascending=[False, True, True]).iloc[0]
        best_key = (
            cohort,
            str(best["prompt_variant"]),
            _condition_key_from_table_row(best),
            str(best["model"]),
        )
        predictions = prediction_store.get(best_key)
        if predictions is None or predictions.empty:
            continue

        precision, recall, f1, support = precision_recall_fscore_support(
            predictions["label"],
            predictions["pred"],
            labels=MACRO_LABELS,
            average=None,
            zero_division=0,
        )
        best_rows.append(
            {
                "cohort": cohort,
                "prompt_variant": best["prompt_variant"],
                "model": best["model"],
                "regime": best["regime"],
                "prompt_regime": best["prompt_regime"],
                "macro_f1": best["macro_f1"],
            }
        )
        for idx, label in enumerate(MACRO_LABELS):
            table_2_rows.append(
                {
                    "cohort": cohort,
                    "model": best["model"],
                    "regime": best["regime"],
                    "prompt_regime": best["prompt_regime"],
                    "class": label,
                    "precision": float(precision[idx]),
                    "recall": float(recall[idx]),
                    "f1": float(f1[idx]),
                    "support": int(support[idx]),
                }
            )

    table_2 = pd.DataFrame(
        table_2_rows,
        columns=["cohort", "model", "regime", "prompt_regime", "class", "precision", "recall", "f1", "support"],
    )
    best_configs = pd.DataFrame(best_rows)
    return table_2, best_configs


def build_consolidated_tables(
    *,
    variant_roots: Sequence[Tuple[str, str, str]] = DEFAULT_VARIANT_ROOTS,
    primary_m: int = PRIMARY_M,
    bootstrap_reps: int = 10000,
    alpha: float = 0.05,
    random_state: int = 0,
    force_recalc_regex: bool = True,
    strict: bool = False,
) -> Dict[str, object]:
    specs = discover_variant_specs(variant_roots)
    warnings: List[str] = []
    ensure_regex_artifacts(specs, force_recalc=force_recalc_regex, warnings=warnings)

    regex_by_cohort: Dict[str, pd.DataFrame] = {}
    for spec in specs:
        regex_predictions = load_regex_predictions(spec)
        if spec.cohort not in regex_by_cohort and not regex_predictions.empty:
            regex_by_cohort[spec.cohort] = regex_predictions

    table_1_frames: List[pd.DataFrame] = []
    table_4_frames: List[pd.DataFrame] = []
    prediction_store: Dict[Tuple[str, str, str, str], pd.DataFrame] = {}
    condition_status_rows: List[Dict[str, object]] = []

    for spec in specs:
        gold_df = load_gold_labels(spec.gold_path)
        source_frames = load_condition_source_frames(spec)
        merged_source_frames = {
            source_key: merge_with_gold(frame, gold_df)
            for source_key, frame in source_frames.items()
        }

        for condition in CONDITION_SPECS:
            merged_primary, is_complete = _filter_primary_condition(
                merged_source_frames.get(condition.source_key, pd.DataFrame()),
                condition,
                primary_m=primary_m,
                expected_seeds=EXPECTED_SEEDS,
            )
            available_seeds = sorted(
                merged_primary["seed"].dropna().astype(int).unique().tolist()
            ) if "seed" in merged_primary.columns and not merged_primary.empty else []
            condition_status_rows.append(
                {
                    "cohort": spec.cohort,
                    "prompt_variant": spec.prompt_variant,
                    "condition_key": condition.key,
                    "source_key": condition.source_key,
                    "available_primary_seeds": available_seeds,
                    "is_complete_primary": bool(is_complete),
                    "primary_rows": int(len(merged_primary)),
                }
            )

            if condition.zero_shot:
                condition_ok = not merged_primary.empty
            else:
                condition_ok = bool(is_complete)

            if not condition_ok:
                if merged_primary.empty:
                    warnings.append(
                        f"Missing primary data for {spec.cohort} {spec.prompt_variant} {condition.key}."
                    )
                else:
                    warnings.append(
                        "Incomplete five-seed coverage for "
                        f"{spec.cohort} {spec.prompt_variant} {condition.key}: {available_seeds}"
                    )
                table_1_frames.append(_empty_table_1_rows(spec, condition))
            else:
                table_1_chunk, frames_by_model = _compute_table_1_rows(
                    spec,
                    condition,
                    merged_primary,
                    bootstrap_reps=bootstrap_reps,
                    alpha=alpha,
                    random_state=random_state,
                )
                table_1_frames.append(table_1_chunk)
                for model_name, predictions in frames_by_model.items():
                    prediction_store[(spec.cohort, spec.prompt_variant, condition.key, str(model_name))] = predictions

            merged_all_m = merged_source_frames.get(condition.source_key, pd.DataFrame())
            table_4_frames.append(_compute_table_4_rows(spec, condition, merged_all_m))

    for cohort, regex_predictions in sorted(regex_by_cohort.items()):
        table_1_frames.append(
            _regex_row_for_cohort(
                cohort,
                regex_predictions,
                bootstrap_reps=bootstrap_reps,
                alpha=alpha,
                random_state=_stable_seed(cohort, REGEX_MODEL_NAME, base=random_state),
            )
        )
        prediction_store[(cohort, "short", "regex", REGEX_MODEL_NAME)] = regex_predictions
        prediction_store[(cohort, "long", "regex", REGEX_MODEL_NAME)] = regex_predictions

    table_1 = pd.concat(table_1_frames, ignore_index=True) if table_1_frames else pd.DataFrame()
    nonempty_table_4_frames = [frame for frame in table_4_frames if frame is not None and not frame.empty]
    table_4 = pd.concat(nonempty_table_4_frames, ignore_index=True) if nonempty_table_4_frames else pd.DataFrame()
    condition_status = pd.DataFrame(condition_status_rows)

    if strict and warnings:
        raise RuntimeError("\n".join(warnings))

    table_1 = table_1[
        ["cohort", "prompt_variant", "model", "regime", "prompt_regime", "macro_f1", "ci_low", "ci_high", "n_docs"]
    ].copy()

    table_2_with_committee, table_2_best_configs_with_committee = _build_table_2_variant(
        table_1,
        prediction_store,
        include_committee=True,
    )
    table_2_without_committee, table_2_best_configs_without_committee = _build_table_2_variant(
        table_1,
        prediction_store,
        include_committee=False,
    )

    table_3_rows: List[Dict[str, object]] = []
    comparison_specs = [
        ("label_only_WithUpdate vs zero_shot", "label_only_with_update", "zero_shot"),
        ("rationale_WithUpdate vs label_only_WithUpdate", "rationale_with_update", "label_only_with_update"),
        ("rationale_WithUpdate vs rationale_Random", "rationale_with_update", "rationale_random"),
        ("best_ICL vs regex_baseline", "best", "regex"),
    ]
    for spec in specs:
        for model_name in list(spec.models) + [COMMITTEE_MODEL_NAME]:
            for comparison, left_key, right_key in comparison_specs:
                left_frame = None
                right_frame = None
                if left_key == "best":
                    best_pick = _best_condition_for_model(
                        table_1,
                        cohort=spec.cohort,
                        prompt_variant=spec.prompt_variant,
                        model_name=model_name,
                    )
                    if best_pick is not None:
                        best_condition_key = _condition_key_from_values(*best_pick)
                        left_frame = prediction_store.get((spec.cohort, spec.prompt_variant, best_condition_key, model_name))
                else:
                    left_frame = prediction_store.get((spec.cohort, spec.prompt_variant, left_key, model_name))

                if right_key == "regex":
                    right_frame = prediction_store.get((spec.cohort, spec.prompt_variant, "regex", REGEX_MODEL_NAME))
                else:
                    right_frame = prediction_store.get((spec.cohort, spec.prompt_variant, right_key, model_name))

                comparison_stats = paired_bootstrap_comparison(
                    left_frame if isinstance(left_frame, pd.DataFrame) else pd.DataFrame(),
                    right_frame if isinstance(right_frame, pd.DataFrame) else pd.DataFrame(),
                    B=bootstrap_reps,
                    alpha=alpha,
                    random_state=_stable_seed(
                        spec.cohort,
                        spec.prompt_variant,
                        model_name,
                        comparison,
                        base=random_state,
                    ),
                )
                table_3_rows.append(
                    {
                        "cohort": spec.cohort,
                        "prompt_variant": spec.prompt_variant,
                        "model": model_name,
                        "comparison": comparison,
                        "delta_f1": comparison_stats["delta_f1"],
                        "bootstrap_ci_low": comparison_stats["bootstrap_ci_low"],
                        "bootstrap_ci_high": comparison_stats["bootstrap_ci_high"],
                        "bootstrap_p": comparison_stats["bootstrap_p"],
                        "significant": bool(comparison_stats["bootstrap_p"] < alpha)
                        if pd.notna(comparison_stats["bootstrap_p"])
                        else False,
                    }
                )
    table_3 = pd.DataFrame(
        table_3_rows,
        columns=[
            "cohort",
            "prompt_variant",
            "model",
            "comparison",
            "delta_f1",
            "bootstrap_ci_low",
            "bootstrap_ci_high",
            "bootstrap_p",
            "significant",
        ],
    )

    base_models = list(specs[0].models) if specs else []
    summary_source = table_1[table_1["model"].isin(base_models)].copy() if base_models else pd.DataFrame()
    if not summary_source.empty:
        summary_source = summary_source.dropna(subset=["macro_f1"])
    if not summary_source.empty:
        table_5 = (
            summary_source.groupby(["cohort", "prompt_variant", "regime", "prompt_regime"], dropna=False)["macro_f1"]
            .agg(mean_macro_f1="mean", median_macro_f1="median", min_macro_f1="min", max_macro_f1="max")
            .reset_index()
        )
    else:
        table_5 = pd.DataFrame(
            columns=[
                "cohort",
                "prompt_variant",
                "regime",
                "prompt_regime",
                "mean_macro_f1",
                "median_macro_f1",
                "min_macro_f1",
                "max_macro_f1",
            ]
        )

    expected_summary_rows = []
    for spec in specs:
        for condition in CONDITION_SPECS:
            expected_summary_rows.append(
                {
                    "cohort": spec.cohort,
                    "prompt_variant": spec.prompt_variant,
                    "regime": condition.regime,
                    "prompt_regime": condition.prompt_regime,
                }
            )
    if expected_summary_rows:
        expected_table_5 = pd.DataFrame(expected_summary_rows)
        expected_table_5["_regime_key"] = expected_table_5["regime"].fillna("__NA__")
        table_5["_regime_key"] = table_5["regime"].fillna("__NA__")
        table_5 = (
            expected_table_5.merge(
                table_5.drop(columns=["regime"]),
                on=["cohort", "prompt_variant", "prompt_regime", "_regime_key"],
                how="left",
            )
            .drop(columns=["_regime_key"])
        )

    return {
        "table_1": table_1,
        "table_2": table_2_with_committee,
        "table_2_best_configs": table_2_best_configs_with_committee,
        "table_2_with_committee": table_2_with_committee,
        "table_2_best_configs_with_committee": table_2_best_configs_with_committee,
        "table_2_without_committee": table_2_without_committee,
        "table_2_best_configs_without_committee": table_2_best_configs_without_committee,
        "table_3": table_3,
        "table_4": table_4,
        "table_5": table_5,
        "condition_status": condition_status,
        "warnings": warnings,
    }


def _condition_key_from_values(regime: str, prompt_regime: str) -> str:
    if prompt_regime == PROMPT_REGIME_ZERO_SHOT:
        return "zero_shot"
    if regime == "WithUpdate" and prompt_regime == PROMPT_REGIME_LABEL_ONLY:
        return "label_only_with_update"
    if regime == "WithUpdate" and prompt_regime == PROMPT_REGIME_RATIONALE:
        return "rationale_with_update"
    if regime == "Random" and prompt_regime == PROMPT_REGIME_LABEL_ONLY:
        return "label_only_random"
    if regime == "Random" and prompt_regime == PROMPT_REGIME_RATIONALE:
        return "rationale_random"
    raise KeyError(f"Unsupported condition values: regime={regime}, prompt_regime={prompt_regime}")


def _condition_key_from_table_row(row: pd.Series) -> str:
    return _condition_key_from_values(row["regime"], row["prompt_regime"])


def save_consolidated_tables(artifacts: Mapping[str, object], output_dir: Path) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: Dict[str, Path] = {}
    for key in [
        "table_1",
        "table_2",
        "table_2_best_configs",
        "table_2_with_committee",
        "table_2_best_configs_with_committee",
        "table_2_without_committee",
        "table_2_best_configs_without_committee",
        "table_3",
        "table_4",
        "table_5",
        "condition_status",
    ]:
        value = artifacts.get(key)
        if isinstance(value, pd.DataFrame):
            path = output_dir / f"{key}.csv"
            value.to_csv(path, index=False)
            saved[key] = path
    warnings_path = output_dir / "warnings.json"
    warnings_path.write_text(json.dumps(list(artifacts.get("warnings", [])), indent=2), encoding="utf-8")
    saved["warnings"] = warnings_path
    return saved
