#!/usr/bin/env python3
"""Run balanced-draw simulations plus paired statistical tests for reporting.

This script keeps the notebook light:
- reuses scripts/run_balanced_draw_selection_stats.py to materialize per-target stats
- computes paired full-range comparisons after averaging over seeds
- exports compact CSV/JSON artifacts for notebook plotting
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TARGET_REGIME = "reveal_one_by_one_yhat_round_robin_prob_update"
REGIMES = [
    "pure_random",
    "reveal_one_by_one_yhat_round_robin",
    "reveal_one_by_one_yhat_round_robin_prob_update",
    "reveal_one_by_one_yhat_round_robin_prob_no_update",
]
COMPARATORS = [r for r in REGIMES if r != TARGET_REGIME]
DEFAULT_VARIANTS = {
    "short": {
        "output_root": "results/mimic_streamlined_pipeline_small_models_upd_short",
        "base_predictions": "results/mimic_streamlined_pipeline_small_models_upd_short/zero_shot/all_models.pkl",
    },
    "long": {
        "output_root": "results/mimic_streamlined_pipeline_small_models_upd_long",
        "base_predictions": "results/mimic_streamlined_pipeline_small_models_upd_long/zero_shot/all_models.pkl",
    },
    "indic_short": {
        "output_root": "results/mimic_streamlined_pipeline_small_models_indic_upd_short",
        "base_predictions": "results/mimic_streamlined_pipeline_small_models_indic_upd_short/zero_shot/all_models.pkl",
    },
    "indic_long": {
        "output_root": "results/mimic_streamlined_pipeline_small_models_indic_upd_long",
        "base_predictions": "results/mimic_streamlined_pipeline_small_models_indic_upd_long/zero_shot/all_models.pkl",
    },
}


@dataclass(frozen=True)
class VariantSpec:
    name: str
    output_root: Path
    base_predictions: Path

    @property
    def analysis_dir(self) -> Path:
        return self.output_root


def _parse_int_list(value: Optional[str]) -> Optional[List[int]]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return [int(chunk.strip()) for chunk in text.split(",") if chunk.strip()]


def _parse_str_list(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return [chunk.strip() for chunk in text.split(",") if chunk.strip()]


def _holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(list(p_values), dtype=float)
    n = len(p)
    if n == 0:
        return np.array([], dtype=float)
    order = np.argsort(p)
    ordered = p[order]
    adjusted = np.empty(n, dtype=float)
    running = 0.0
    for i, value in enumerate(ordered):
        factor = n - i
        running = max(running, value * factor)
        adjusted[i] = min(1.0, running)
    out = np.empty(n, dtype=float)
    out[order] = adjusted
    return out


def _wilcoxon_greater(diffs: pd.Series) -> float:
    values = pd.Series(diffs, dtype=float).dropna()
    nonzero = values[values != 0.0]
    if nonzero.empty:
        return 1.0
    return float(
        wilcoxon(
            nonzero,
            alternative="greater",
            zero_method="wilcox",
            correction=False,
            mode="auto",
        ).pvalue
    )


def _bootstrap_mean_median_win(
    diffs: pd.Series,
    *,
    B: int,
    random_state: int,
    ci: float,
) -> Dict[str, float]:
    values = pd.Series(diffs, dtype=float).dropna().to_numpy()
    if values.size == 0:
        return {
            "mean_ci_low": np.nan,
            "mean_ci_high": np.nan,
            "median_ci_low": np.nan,
            "median_ci_high": np.nan,
            "win_rate_ci_low": np.nan,
            "win_rate_ci_high": np.nan,
        }
    rng = np.random.default_rng(random_state)
    alpha = (1.0 - float(ci)) / 2.0
    mean_draws: List[np.ndarray] = []
    median_draws: List[np.ndarray] = []
    win_draws: List[np.ndarray] = []
    remaining = int(B)
    chunk_size = 512
    n = len(values)
    while remaining > 0:
        take = min(chunk_size, remaining)
        idx = rng.integers(0, n, size=(take, n))
        sample = values[idx]
        mean_draws.append(sample.mean(axis=1))
        median_draws.append(np.median(sample, axis=1))
        win_draws.append((sample > 0.0).mean(axis=1))
        remaining -= take
    means = np.concatenate(mean_draws)
    medians = np.concatenate(median_draws)
    wins = np.concatenate(win_draws)
    return {
        "mean_ci_low": float(np.quantile(means, alpha)),
        "mean_ci_high": float(np.quantile(means, 1.0 - alpha)),
        "median_ci_low": float(np.quantile(medians, alpha)),
        "median_ci_high": float(np.quantile(medians, 1.0 - alpha)),
        "win_rate_ci_low": float(np.quantile(wins, alpha)),
        "win_rate_ci_high": float(np.quantile(wins, 1.0 - alpha)),
    }


def _paired_sign_flip_pvalue(
    diffs: pd.Series,
    *,
    B: int,
    random_state: int,
) -> float:
    values = pd.Series(diffs, dtype=float).dropna().to_numpy()
    if values.size == 0:
        return 1.0
    obs = float(values.mean())
    if np.isclose(obs, 0.0):
        return 1.0
    rng = np.random.default_rng(random_state)
    exceed = 0
    draws = 0
    remaining = int(B)
    chunk_size = 512
    while remaining > 0:
        take = min(chunk_size, remaining)
        signs = rng.choice(np.array([-1.0, 1.0], dtype=float), size=(take, values.size))
        null_means = (signs * values).mean(axis=1)
        exceed += int((null_means >= obs).sum())
        draws += take
        remaining -= take
    return float((exceed + 1) / (draws + 1))


def _summarize_diffs(
    diffs: pd.Series,
    *,
    bootstrap_B: int,
    signflip_B: int,
    random_state: int,
    ci: float,
) -> Dict[str, float]:
    values = pd.Series(diffs, dtype=float).dropna()
    out = {
        "n_pairs": int(len(values)),
        "mean_diff": float(values.mean()) if len(values) else np.nan,
        "median_diff": float(values.median()) if len(values) else np.nan,
        "pct_positive": float((values > 0.0).mean()) if len(values) else np.nan,
        "pct_negative": float((values < 0.0).mean()) if len(values) else np.nan,
        "pct_zero": float((values == 0.0).mean()) if len(values) else np.nan,
        "wilcoxon_p_greater": _wilcoxon_greater(values),
        "sign_flip_p_greater": _paired_sign_flip_pvalue(
            values,
            B=signflip_B,
            random_state=random_state,
        ),
    }
    out.update(
        _bootstrap_mean_median_win(
            values,
            B=bootstrap_B,
            random_state=random_state,
            ci=ci,
        )
    )
    return out


def _curve_summary(seed_avg_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (regime, target_model, target_doc), group in seed_avg_df.groupby(
        ["regime", "target_model", "target_doc"],
        sort=True,
    ):
        work = group.sort_values("k")
        x = work["k"].to_numpy(dtype=float)
        y = work["selected_over_annotated_ratio"].to_numpy(dtype=float)
        if len(x) > 1:
            auc = float(np.trapz(y, x=x) / (x[-1] - x[0]))
        else:
            auc = float(y.mean())
        rows.append(
            {
                "regime": regime,
                "target_model": target_model,
                "target_doc": target_doc,
                "mean_ratio_over_k": float(y.mean()),
                "normalized_auc_ratio": auc,
                "min_ratio_over_k": float(y.min()),
                "max_ratio_over_k": float(y.max()),
                "n_k": int(len(work)),
            }
        )
    return pd.DataFrame(rows)


def _seed_average(stats_df: pd.DataFrame) -> pd.DataFrame:
    return (
        stats_df.groupby(
            ["regime", "k", "target_model", "target_doc"],
            as_index=False,
        )["selected_over_annotated_ratio"]
        .mean()
        .sort_values(["regime", "k", "target_model", "target_doc"])
        .reset_index(drop=True)
    )


def _regime_full_range_summary(curve_df: pd.DataFrame) -> pd.DataFrame:
    return (
        curve_df.groupby("regime", as_index=False)
        .agg(
            mean_ratio_over_k=("mean_ratio_over_k", "mean"),
            median_ratio_over_k=("mean_ratio_over_k", "median"),
            mean_normalized_auc_ratio=("normalized_auc_ratio", "mean"),
            median_normalized_auc_ratio=("normalized_auc_ratio", "median"),
            n_pairs=("target_doc", "count"),
        )
        .sort_values("mean_ratio_over_k", ascending=False)
        .reset_index(drop=True)
    )


def _primary_comparisons(
    curve_df: pd.DataFrame,
    *,
    variant: str,
    bootstrap_B: int,
    signflip_B: int,
    random_state: int,
    ci: float,
) -> pd.DataFrame:
    pivot = curve_df.pivot_table(
        index=["target_model", "target_doc"],
        columns="regime",
        values="mean_ratio_over_k",
    ).dropna()
    rows: List[Dict[str, object]] = []
    for idx, other in enumerate(COMPARATORS):
        diffs = pivot[TARGET_REGIME] - pivot[other]
        row = {
            "variant": variant,
            "target_regime": TARGET_REGIME,
            "other_regime": other,
            "target_mean_endpoint": float(pivot[TARGET_REGIME].mean()),
            "other_mean_endpoint": float(pivot[other].mean()),
        }
        row.update(
            _summarize_diffs(
                diffs,
                bootstrap_B=bootstrap_B,
                signflip_B=signflip_B,
                random_state=random_state + idx,
                ci=ci,
            )
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    out["wilcoxon_p_holm"] = _holm_adjust(out["wilcoxon_p_greater"].tolist())
    out["wilcoxon_reject_holm"] = out["wilcoxon_p_holm"] <= 0.05
    out["sign_flip_p_holm"] = _holm_adjust(out["sign_flip_p_greater"].tolist())
    out["sign_flip_reject_holm"] = out["sign_flip_p_holm"] <= 0.05
    return out.sort_values("mean_diff", ascending=False).reset_index(drop=True)


def _per_k_comparisons(
    seed_avg_df: pd.DataFrame,
    *,
    variant: str,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for k, group in seed_avg_df.groupby("k", sort=True):
        pivot = group.pivot_table(
            index=["target_model", "target_doc"],
            columns="regime",
            values="selected_over_annotated_ratio",
        ).dropna()
        for other in COMPARATORS:
            diffs = pivot[TARGET_REGIME] - pivot[other]
            rows.append(
                {
                    "variant": variant,
                    "k": int(k),
                    "target_regime": TARGET_REGIME,
                    "other_regime": other,
                    "n_pairs": int(len(diffs)),
                    "mean_diff": float(diffs.mean()),
                    "median_diff": float(diffs.median()),
                    "pct_positive": float((diffs > 0.0).mean()),
                    "pct_negative": float((diffs < 0.0).mean()),
                    "pct_zero": float((diffs == 0.0).mean()),
                    "wilcoxon_p_greater": _wilcoxon_greater(diffs),
                }
            )
    out = pd.DataFrame(rows).sort_values(["other_regime", "k"]).reset_index(drop=True)
    out["wilcoxon_p_holm"] = _holm_adjust(out["wilcoxon_p_greater"].tolist())
    out["wilcoxon_reject_holm"] = out["wilcoxon_p_holm"] <= 0.05
    return out


def _model_robustness(curve_df: pd.DataFrame, *, variant: str) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for target_model, group in curve_df.groupby("target_model", sort=True):
        pivot = group.pivot_table(
            index="target_doc",
            columns="regime",
            values="mean_ratio_over_k",
        ).dropna()
        for other in COMPARATORS:
            diffs = pivot[TARGET_REGIME] - pivot[other]
            rows.append(
                {
                    "variant": variant,
                    "target_model": target_model,
                    "target_regime": TARGET_REGIME,
                    "other_regime": other,
                    "n_pairs": int(len(diffs)),
                    "mean_diff": float(diffs.mean()),
                    "median_diff": float(diffs.median()),
                    "pct_positive": float((diffs > 0.0).mean()),
                    "pct_negative": float((diffs < 0.0).mean()),
                    "pct_zero": float((diffs == 0.0).mean()),
                    "wilcoxon_p_greater": _wilcoxon_greater(diffs),
                }
            )
    out = pd.DataFrame(rows).sort_values(["other_regime", "target_model"]).reset_index(drop=True)
    out["wilcoxon_p_holm"] = _holm_adjust(out["wilcoxon_p_greater"].tolist())
    out["wilcoxon_reject_holm"] = out["wilcoxon_p_holm"] <= 0.05
    return out


def _variant_specs_from_args(args: argparse.Namespace) -> List[VariantSpec]:
    names = _parse_str_list(args.variants)
    if names is None:
        names = list(DEFAULT_VARIANTS.keys())
    missing = [name for name in names if name not in DEFAULT_VARIANTS]
    if missing:
        raise KeyError(f"Unknown variants: {missing}. Available: {sorted(DEFAULT_VARIANTS)}")
    specs = []
    for name in names:
        meta = DEFAULT_VARIANTS[name]
        specs.append(
            VariantSpec(
                name=name,
                output_root=(PROJECT_ROOT / meta["output_root"]).resolve(),
                base_predictions=(PROJECT_ROOT / meta["base_predictions"]).resolve(),
            )
        )
    return specs


def _load_cfg(config_path: Path) -> Mapping[str, object]:
    return json.loads(config_path.read_text())


def _resolve_seeds(cfg: Mapping[str, object], override: Optional[List[int]]) -> List[int]:
    if override is not None:
        return [int(x) for x in override]
    loo_cfg = dict(cfg.get("loo", {}))
    return [int(x) for x in loo_cfg.get("seeds", [0, 30, 500, 6370, 17893])]


def _resolve_k_values(override: Optional[List[int]]) -> List[int]:
    if override is not None:
        return [int(x) for x in override]
    return list(range(2, 21))


def _run_balanced_draw_simulation(
    *,
    config_path: Path,
    variant: VariantSpec,
    analysis_subdir: str,
    dataset: str,
    seeds: Sequence[int],
    k_values: Sequence[int],
    force_recompute: bool,
) -> None:
    cmd = [
        sys.executable,
        str((PROJECT_ROOT / "scripts" / "run_balanced_draw_selection_stats.py").resolve()),
        "--config",
        str(config_path.resolve()),
        "--analysis-subdir",
        str(analysis_subdir),
        "--output-root",
        str(variant.output_root),
        "--base-predictions",
        str(variant.base_predictions),
        "--dataset",
        dataset,
        "--seeds",
        ",".join(str(int(x)) for x in seeds),
        "--k-values",
        ",".join(str(int(x)) for x in k_values),
        "--regimes",
        ",".join(REGIMES),
    ]
    if force_recompute:
        cmd.append("--force-recompute")
    print(f"[run] {variant.name}: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def _write_metadata(
    out_dir: Path,
    *,
    variant: VariantSpec,
    config_path: Path,
    seeds: Sequence[int],
    k_values: Sequence[int],
    analysis_subdir: str,
    bootstrap_B: int,
    signflip_B: int,
    ci: float,
) -> None:
    payload = {
        "variant": variant.name,
        "config_path": str(config_path.resolve()),
        "output_root": str(variant.output_root),
        "base_predictions": str(variant.base_predictions),
        "analysis_subdir": analysis_subdir,
        "target_regime": TARGET_REGIME,
        "regimes": REGIMES,
        "comparators": COMPARATORS,
        "seeds": [int(x) for x in seeds],
        "k_values": [int(x) for x in k_values],
        "primary_endpoint": "mean_ratio_over_k",
        "descriptive_endpoint": "normalized_auc_ratio",
        "bootstrap_reps": int(bootstrap_B),
        "sign_flip_reps": int(signflip_B),
        "ci_level": float(ci),
    }
    (out_dir / "inference_metadata.json").write_text(json.dumps(payload, indent=2))


def _run_inference_for_variant(
    *,
    variant: VariantSpec,
    analysis_subdir: str,
    bootstrap_B: int,
    signflip_B: int,
    random_state: int,
    ci: float,
) -> None:
    out_dir = variant.output_root / analysis_subdir
    stats_path = out_dir / "balanced_draw_stats_per_target.csv"
    agg_k_path = out_dir / "balanced_draw_stats_agg_k.csv"
    agg_seed_k_path = out_dir / "balanced_draw_stats_agg_seed_k.csv"
    stats_df = pd.read_csv(stats_path, low_memory=False)
    agg_k_df = pd.read_csv(agg_k_path)
    agg_seed_k_df = pd.read_csv(agg_seed_k_path)

    stats_df = stats_df[stats_df["regime"].astype(str).isin(REGIMES)].copy()
    agg_k_df = agg_k_df[agg_k_df["regime"].astype(str).isin(REGIMES)].copy()
    agg_seed_k_df = agg_seed_k_df[agg_seed_k_df["regime"].astype(str).isin(REGIMES)].copy()

    seed_avg_df = _seed_average(stats_df)
    curve_df = _curve_summary(seed_avg_df)
    regime_summary_df = _regime_full_range_summary(curve_df)
    primary_df = _primary_comparisons(
        curve_df,
        variant=variant.name,
        bootstrap_B=bootstrap_B,
        signflip_B=signflip_B,
        random_state=random_state,
        ci=ci,
    )
    per_k_df = _per_k_comparisons(
        seed_avg_df,
        variant=variant.name,
    )
    model_df = _model_robustness(
        curve_df,
        variant=variant.name,
    )

    seed_avg_df.to_csv(out_dir / "balanced_draw_stats_seed_averaged_per_target_k.csv", index=False)
    curve_df.to_csv(out_dir / "balanced_draw_stats_full_range_curve_summary.csv", index=False)
    regime_summary_df.to_csv(out_dir / "balanced_draw_stats_regime_full_range_summary.csv", index=False)
    primary_df.to_csv(out_dir / "balanced_draw_stats_primary_mean_over_k_comparisons.csv", index=False)
    per_k_df.to_csv(out_dir / "balanced_draw_stats_per_k_wilcoxon_comparisons.csv", index=False)
    model_df.to_csv(out_dir / "balanced_draw_stats_model_robustness_mean_over_k.csv", index=False)
    print(f"[wrote] {variant.name}: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run balanced-draw simulations and paired inference for short/long zero-shot outputs.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/mimic_baseline_loo_config.json",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default="short,long",
        help="Comma-separated variants. Available: short,long,indic_short,indic_long",
    )
    parser.add_argument(
        "--analysis-subdir",
        type=str,
        default="analysis/ping_pong_balanced_draw_stats_inference",
    )
    parser.add_argument("--dataset", type=str, default="mimic")
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--k-values", type=str, default=None)
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument("--sign-flip-reps", type=int, default=10000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--ci-level", type=float, default=0.95)
    parser.add_argument("--force-recompute", action="store_true")
    args = parser.parse_args()

    config_path = (PROJECT_ROOT / args.config).resolve()
    cfg = _load_cfg(config_path)
    seeds = _resolve_seeds(cfg, _parse_int_list(args.seeds))
    k_values = _resolve_k_values(_parse_int_list(args.k_values))
    variants = _variant_specs_from_args(args)

    print(f"config={config_path}")
    print(f"variants={[v.name for v in variants]}")
    print(f"seeds={seeds}")
    print(f"k_values={k_values}")
    print(f"analysis_subdir={args.analysis_subdir}")
    print(f"bootstrap_reps={args.bootstrap_reps}")
    print(f"sign_flip_reps={args.sign_flip_reps}")

    for variant in variants:
        _run_balanced_draw_simulation(
            config_path=config_path,
            variant=variant,
            analysis_subdir=args.analysis_subdir,
            dataset=str(args.dataset),
            seeds=seeds,
            k_values=k_values,
            force_recompute=bool(args.force_recompute),
        )
        out_dir = variant.output_root / args.analysis_subdir
        _write_metadata(
            out_dir,
            variant=variant,
            config_path=config_path,
            seeds=seeds,
            k_values=k_values,
            analysis_subdir=args.analysis_subdir,
            bootstrap_B=int(args.bootstrap_reps),
            signflip_B=int(args.sign_flip_reps),
            ci=float(args.ci_level),
        )
        _run_inference_for_variant(
            variant=variant,
            analysis_subdir=args.analysis_subdir,
            bootstrap_B=int(args.bootstrap_reps),
            signflip_B=int(args.sign_flip_reps),
            random_state=int(args.random_state),
            ci=float(args.ci_level),
        )


if __name__ == "__main__":
    main()
