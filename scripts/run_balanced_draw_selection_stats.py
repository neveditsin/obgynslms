#!/usr/bin/env python3
"""Simulate balanced exemplar collection without compensation turns.

This script mirrors the regime names used in notebooks/ping_pong_selection_stats.ipynb
but changes the process:
- no compensation turns
- draw one document at a time from the remaining pool
- reveal gold after each draw
- keep drawing until we collect k exemplars per gold class (or exhaust the pool)

Primary metric:
    selected_over_annotated_ratio = selected_exemplars / annotated_total
Ideal value is 1.0 (no useless annotations).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_documents
from src.loo_selector import prepare_loo_resources

try:
    import torch
except Exception:
    torch = None


def _parse_int_list(value: Optional[str]) -> Optional[List[int]]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    out: List[int] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        out.append(int(chunk))
    return out


def _parse_str_list(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return [c.strip() for c in text.split(",") if c.strip()]


def _canon_label(label: Optional[str]) -> Optional[str]:
    if label is None:
        return None
    key = str(label).strip().lower().rstrip(".")
    if key in {"early", "early pregnancy", "early active pregnancy"}:
        return "early"
    if key in {"late", "late pregnancy", "late active pregnancy"}:
        return "late"
    if key in {
        "unrelated",
        "unrelated or no current pregnancy",
        "no current pregnancy",
        "no active pregnancy",
        "no",
        "none",
    }:
        return "unrelated"
    return key


def _gold_sort_key(label: str) -> Tuple[int, str]:
    canon = _canon_label(label)
    rank = {"early": 0, "late": 1, "unrelated": 2}.get(canon, 99)
    return (rank, str(label))


def _score_with_mode(mode: str, h: float, d: float, *, has_anchor: bool, alpha: float = 0.5) -> float:
    if not has_anchor:
        return float(h)
    if mode == "harmonic":
        denom = (h + d) if (h + d) > 0 else 1e-12
        return float((2.0 * h * d) / denom)
    if mode == "sum":
        return float(alpha) * float(h) + (1.0 - float(alpha)) * float(d)
    if mode == "product":
        return float(h) * float(d)
    if mode == "uncertainty":
        return float(h)
    if mode == "diversity":
        return float(d)
    raise ValueError("score mode must be one of {'harmonic','sum','product','uncertainty','diversity'}")


def build_regime_specs_no_compensation() -> Dict[str, Dict[str, Any]]:
    """Regimes for balanced-draw no-comp simulation.

    We keep:
    - pure_random baseline
    - pp2 as runner-default-like ping policy (global pool + round-local diversity + reveal 1-by-1)
    - reveal_one_by_one_yhat_round_robin: reveal mode that targets predicted y-hat
      classes in strict round-robin order (c1 -> c2 -> c3 -> ...), with no
      uncertainty/diversity scoring inside the selected bucket.
    - reveal_one_by_one_yhat_round_robin_uncertainty_weighted: same round-robin
      targeting, but sample inside each predicted bucket proportional to vote entropy.
    - reveal_one_by_one_yhat_round_robin_prob_update: round-robin target classes,
      sample docs by soft class probabilities from committee vote counts, and on
      mismatch reassign probability mass in a similarity neighborhood.
    - reveal_one_by_one_yhat_round_robin_prob_no_update: same as
      reveal_one_by_one_yhat_round_robin_prob_update, but without mismatch-driven
      probability redistribution.
    - a6_granule_prior_round_robin_no_pong: granule-prior guided round-robin
      targeting (mirrors runner a6 ping ablation), no compensation turns.
    """
    common = {
        "score": "harmonic",
        "use_pred_round_robin": True,
        "global_pool_round_local_diversity_after_first": False,
        "reveal_one_by_one": False,
        "reveal_ping_pred_round_robin": False,
        "reveal_ping_pred_round_robin_uncertainty_weighted": False,
        "reveal_ping_pred_round_robin_prob_update": False,
        "reveal_ping_pred_round_robin_prob_no_update": False,
        "reveal_ping_pred_deficit_h": False,
        "reveal_ping_random_pool": False,
        "reveal_ping_alternate_uncertainty_diversity": False,
        "granule_prior_round_robin": False,
        "granule_prior_cap": 0.95,
    }

    return {
        "pure_random": {
            "version": 1,
            "kwargs": {
                "pure_random": True,
            },
        },
        "pp2": {
            "version": 1,
            "kwargs": {
                **common,
                "score": "harmonic",
                "use_pred_round_robin": False,
                "global_pool_round_local_diversity_after_first": True,
                "reveal_one_by_one": True,
            },
        },
        "reveal_one_by_one_yhat_round_robin": {
            "version": 1,
            "kwargs": {
                **common,
                "score": "harmonic",
                "use_pred_round_robin": True,
                "reveal_one_by_one": True,
                "reveal_ping_pred_round_robin": True,
            },
        },
        "reveal_one_by_one_yhat_round_robin_uncertainty_weighted": {
            "version": 1,
            "kwargs": {
                **common,
                "score": "harmonic",
                "use_pred_round_robin": True,
                "reveal_one_by_one": True,
                "reveal_ping_pred_round_robin_uncertainty_weighted": True,
            },
        },
        "reveal_one_by_one_yhat_round_robin_prob_update": {
            "version": 1,
            "kwargs": {
                **common,
                "score": "harmonic",
                "use_pred_round_robin": True,
                "reveal_one_by_one": True,
                "reveal_ping_pred_round_robin_prob_update": True,
            },
        },
        "reveal_one_by_one_yhat_round_robin_prob_no_update": {
            "version": 1,
            "kwargs": {
                **common,
                "score": "harmonic",
                "use_pred_round_robin": True,
                "reveal_one_by_one": True,
                "reveal_ping_pred_round_robin_prob_no_update": True,
            },
        },
        "a6_granule_prior_round_robin_no_pong": {
            "version": 1,
            "kwargs": {
                **common,
                "score": "uncertainty",
                "use_pred_round_robin": False,
                "granule_prior_round_robin": True,
                "granule_prior_cap": 0.95,
            },
        },
    }


def _filter_seed_k_pairs(df: pd.DataFrame, pairs: set[Tuple[int, int]]) -> pd.DataFrame:
    if df is None or df.empty or not pairs:
        return pd.DataFrame(columns=getattr(df, "columns", None))
    keys = list(zip(df["seed"].astype(int), df["k"].astype(int)))
    mask = [key in pairs for key in keys]
    return df.loc[mask].copy()


def _resolve_dataset_paths(cfg: Dict[str, Any], dataset_override: Optional[str]) -> Tuple[str, Path, Path]:
    paths_cfg = dict(cfg.get("paths", {}))
    dataset = str(dataset_override or paths_cfg.get("dataset", "mimic")).strip().lower()
    dataset_paths = paths_cfg.get("dataset_paths", {}) or {}
    entry = dataset_paths.get(dataset, {}) if isinstance(dataset_paths, dict) else {}
    data_dir = Path(str(entry.get("data_dir", paths_cfg.get("data_dir"))))
    gold_path = Path(str(entry.get("gold_labels", paths_cfg.get("gold_labels"))))
    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir not found: {data_dir}")
    if not gold_path.exists():
        raise FileNotFoundError(f"gold_labels not found: {gold_path}")
    return dataset, data_dir, gold_path


def _resolve_output_root(cfg: Dict[str, Any], output_root_override: Optional[str]) -> Path:
    if output_root_override:
        root = Path(str(output_root_override))
    else:
        paths_cfg = dict(cfg.get("paths", {}))
        root = Path(str(paths_cfg.get("output_root", "results/mimic_streamlined_pipeline_small_models")))
    return root


def _resolve_base_predictions_path(
    cfg: Dict[str, Any],
    output_root: Path,
    base_predictions_override: Optional[str],
) -> Path:
    if base_predictions_override:
        path = Path(str(base_predictions_override))
    else:
        paths_cfg = dict(cfg.get("paths", {}))
        cfg_base = paths_cfg.get("base_predictions_path")
        if cfg_base:
            path = Path(str(cfg_base))
        else:
            path = output_root / "zero_shot" / "all_models.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Base predictions not found: {path}")
    return path


def _seed_k_pairs_for_regime(
    regime_name: str,
    *,
    seeds: Sequence[int],
    k_values: Sequence[int],
) -> set[Tuple[int, int]]:
    """Return seed/k workload for a regime.

    pp2 is deterministic in this balanced-draw simulator, so evaluating more than one
    seed is redundant.
    """
    seeds_use = [int(s) for s in seeds]
    if str(regime_name) in {"pp2", "reveal_one_by_one_yhat_round_robin"} and seeds_use:
        seeds_use = [seeds_use[0]]
    return {(int(s), int(k)) for s in seeds_use for k in k_values}


def simulate_balanced_draw_no_comp(
    table: pd.DataFrame,
    *,
    k: int,
    id_col: str,
    pred_col: str,
    entropy_col: str,
    gold_col: str,
    embedding_cache: Dict[str, np.ndarray],
    seed: int,
    regime_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    if k <= 0 or table is None or table.empty:
        return {
            "annotated_total": 0,
            "selected_exemplars": 0,
            "useless_annotated": 0,
            "selected_over_annotated_ratio": 0.0,
            "completion_fraction": 0.0,
            "completed": False,
            "target_exemplars": 0,
            "num_target_classes": 0,
            "pool_size": 0,
            "possible_to_complete": False,
            "accepted_counts_json": "{}",
            "observed_counts_json": "{}",
        }

    base_cols = [id_col, pred_col, entropy_col, gold_col]
    if "prediction_vector" in table.columns:
        base_cols.append("prediction_vector")
    if "counts" in table.columns:
        base_cols.append("counts")
    base = table[base_cols].copy()
    base[id_col] = base[id_col].astype(str)
    base[pred_col] = base[pred_col].astype(str)
    base[gold_col] = base[gold_col].apply(lambda x: None if pd.isna(x) else str(x))
    base = base.reset_index(drop=True)

    ids = base[id_col].tolist()
    pred_labels = base[pred_col].astype(str).tolist()
    gold_labels = [None if g is None else str(g) for g in base[gold_col].tolist()]
    pool_size = len(ids)
    if pool_size == 0:
        return {
            "annotated_total": 0,
            "selected_exemplars": 0,
            "useless_annotated": 0,
            "selected_over_annotated_ratio": 0.0,
            "completion_fraction": 0.0,
            "completed": False,
            "target_exemplars": 0,
            "num_target_classes": 0,
            "pool_size": 0,
            "possible_to_complete": False,
            "accepted_counts_json": "{}",
            "observed_counts_json": "{}",
        }

    target_labels = sorted(
        {str(g) for g in gold_labels if g is not None},
        key=_gold_sort_key,
    )
    if not target_labels:
        return {
            "annotated_total": 0,
            "selected_exemplars": 0,
            "useless_annotated": 0,
            "selected_over_annotated_ratio": 0.0,
            "completion_fraction": 0.0,
            "completed": False,
            "target_exemplars": 0,
            "num_target_classes": 0,
            "pool_size": pool_size,
            "possible_to_complete": False,
            "accepted_counts_json": "{}",
            "observed_counts_json": "{}",
        }

    ent = base[entropy_col].astype(float).to_numpy()
    if ent.size == 0:
        return {
            "annotated_total": 0,
            "selected_exemplars": 0,
            "useless_annotated": 0,
            "selected_over_annotated_ratio": 0.0,
            "completion_fraction": 0.0,
            "completed": False,
            "target_exemplars": int(k) * len(target_labels),
            "num_target_classes": len(target_labels),
            "pool_size": pool_size,
            "possible_to_complete": False,
            "accepted_counts_json": "{}",
            "observed_counts_json": "{}",
        }

    e_min, e_max = float(np.min(ent)), float(np.max(ent))
    if e_max > e_min:
        h_hat = (ent - e_min) / (e_max - e_min)
    else:
        h_hat = np.zeros_like(ent)
    base["_H"] = h_hat

    try:
        embs = np.stack([embedding_cache[fid] for fid in ids], axis=0)
    except KeyError as exc:
        missing = exc.args[0]
        raise KeyError(f"Embedding cache missing vector for '{missing}'") from exc

    rng = np.random.default_rng(int(seed))

    accepted_counts: Dict[str, int] = {g: 0 for g in target_labels}
    observed_counts: Dict[str, int] = {g: 0 for g in target_labels}
    available_counts: Dict[str, int] = {
        g: int(sum(1 for lab in gold_labels if lab == g))
        for g in target_labels
    }
    possible_to_complete = all(int(v) >= int(k) for v in available_counts.values())

    remaining = set(range(pool_size))
    annotated_order: List[int] = []
    accepted_order: List[int] = []
    max_sim_to_annotated = np.zeros(pool_size, dtype=np.float32)
    reveal_max_sim_to_gold: Dict[str, np.ndarray] = {
        str(g): np.zeros(pool_size, dtype=np.float32) for g in target_labels
    }

    score_mode = str(regime_kwargs.get("score", "harmonic")).strip().lower()
    alpha = float(regime_kwargs.get("alpha", 0.5))
    use_pred_round_robin = bool(regime_kwargs.get("use_pred_round_robin", True))
    local_div_after_first = bool(
        regime_kwargs.get("global_pool_round_local_diversity_after_first", False)
    )
    reveal_one_by_one = bool(regime_kwargs.get("reveal_one_by_one", False))
    reveal_ping_pred_round_robin = bool(
        regime_kwargs.get("reveal_ping_pred_round_robin", False)
    )
    reveal_ping_pred_round_robin_uncertainty_weighted = bool(
        regime_kwargs.get("reveal_ping_pred_round_robin_uncertainty_weighted", False)
    )
    reveal_ping_pred_round_robin_prob_update = bool(
        regime_kwargs.get("reveal_ping_pred_round_robin_prob_update", False)
    )
    reveal_ping_pred_round_robin_prob_no_update = bool(
        regime_kwargs.get("reveal_ping_pred_round_robin_prob_no_update", False)
    )
    reveal_ping_pred_deficit_h = bool(regime_kwargs.get("reveal_ping_pred_deficit_h", False))
    reveal_ping_random_pool = bool(regime_kwargs.get("reveal_ping_random_pool", False))
    reveal_ping_alt_ud = bool(
        regime_kwargs.get("reveal_ping_alternate_uncertainty_diversity", False)
    )
    granule_prior_round_robin = bool(regime_kwargs.get("granule_prior_round_robin", False))
    granule_prior_cap = float(regime_kwargs.get("granule_prior_cap", 0.95))
    pure_random = bool(regime_kwargs.get("pure_random", False))

    reveal_overrides = (
        int(reveal_ping_pred_round_robin)
        + int(reveal_ping_pred_round_robin_uncertainty_weighted)
        + int(reveal_ping_pred_round_robin_prob_update)
        + int(reveal_ping_pred_round_robin_prob_no_update)
        + int(reveal_ping_pred_deficit_h)
        + int(reveal_ping_random_pool)
        + int(reveal_ping_alt_ud)
    )
    if reveal_overrides > 1:
        raise ValueError(
            "At most one reveal override may be enabled: "
            "{reveal_ping_pred_round_robin, reveal_ping_pred_round_robin_uncertainty_weighted, "
            "reveal_ping_pred_round_robin_prob_update, "
            "reveal_ping_pred_round_robin_prob_no_update, "
            "reveal_ping_pred_deficit_h, "
            "reveal_ping_random_pool, reveal_ping_alternate_uncertainty_diversity}"
        )

    pred_order = list(base.groupby(pred_col, sort=False).groups.keys())
    pred_pointer = 0

    round_budget = max(1, len(pred_order)) if pred_order else 1
    global_round_picks = 0
    global_round_max_sim = np.zeros(pool_size, dtype=np.float32)
    reveal_component_count = 0
    reveal_pred_pointer = 0

    def _quotas_satisfied() -> bool:
        return all(int(accepted_counts[g]) >= int(k) for g in target_labels)

    def _update_similarity_after_pick(idx: int) -> None:
        v = embs[idx]
        sims = embs @ v
        if remaining:
            r_idx = np.fromiter(remaining, dtype=int)
            max_sim_to_annotated[r_idx] = np.maximum(max_sim_to_annotated[r_idx], sims[r_idx])
            g = gold_labels[idx]
            if g is not None and str(g) in reveal_max_sim_to_gold:
                arr = reveal_max_sim_to_gold[str(g)]
                arr[r_idx] = np.maximum(arr[r_idx], sims[r_idx])

    def _diversity(idx: int) -> float:
        return 1.0 - float(max_sim_to_annotated[idx])

    def _reveal_diversity(idx: int) -> float:
        if not annotated_order:
            return 1.0
        count_vals = list(observed_counts.values())
        if not count_vals:
            return _diversity(idx)
        max_count = max(count_vals)
        min_count = min(count_vals)
        if max_count <= min_count:
            return _diversity(idx)
        overrep = [g for g, c in observed_counts.items() if int(c) == int(max_count)]
        if not overrep:
            return _diversity(idx)
        max_sim = 0.0
        found = False
        for g in overrep:
            arr = reveal_max_sim_to_gold.get(str(g))
            if arr is None:
                continue
            max_sim = max(max_sim, float(arr[idx]))
            found = True
        return (1.0 - max_sim) if found else _diversity(idx)

    def _best_by_score(
        candidates: Sequence[int],
        *,
        mode: str,
        reveal_div_mode: bool,
    ) -> Optional[int]:
        if not candidates:
            return None
        best_i: Optional[int] = None
        best_s = -1.0
        best_h = -1.0
        for i in candidates:
            h = float(base["_H"].iat[i])
            d = _reveal_diversity(i) if reveal_div_mode else _diversity(i)
            s = _score_with_mode(mode, h, d, has_anchor=bool(annotated_order), alpha=alpha)
            if (s > best_s) or (np.isclose(s, best_s) and h > best_h) or (
                np.isclose(s, best_s) and np.isclose(h, best_h) and (best_i is None or i < best_i)
            ):
                best_i = i
                best_s = s
                best_h = h
        return best_i

    def _pick_pred_round_robin() -> int:
        nonlocal pred_pointer
        candidates: List[int] = []
        if pred_order:
            for _ in range(len(pred_order)):
                p = pred_order[pred_pointer]
                pred_pointer = (pred_pointer + 1) % len(pred_order)
                bucket = [i for i in remaining if pred_labels[i] == p]
                if bucket:
                    candidates = bucket
                    break
        if not candidates:
            candidates = list(remaining)
        best = _best_by_score(candidates, mode=score_mode, reveal_div_mode=False)
        if best is not None:
            return int(best)
        return int(rng.choice(candidates))

    def _pick_global_pool() -> int:
        nonlocal global_round_picks, global_round_max_sim
        if global_round_picks >= round_budget:
            global_round_picks = 0
            global_round_max_sim = np.zeros(pool_size, dtype=np.float32)

        candidates = list(remaining)
        best_i: Optional[int] = None
        best_s = -1.0
        best_h = -1.0
        for i in candidates:
            h = float(base["_H"].iat[i])
            if local_div_after_first and global_round_picks > 0:
                d = 1.0 - float(global_round_max_sim[i])
            else:
                d = _diversity(i)
            s = _score_with_mode(score_mode, h, d, has_anchor=bool(annotated_order), alpha=alpha)
            if (s > best_s) or (np.isclose(s, best_s) and h > best_h) or (
                np.isclose(s, best_s) and np.isclose(h, best_h) and (best_i is None or i < best_i)
            ):
                best_i = i
                best_s = s
                best_h = h

        pick = int(rng.choice(candidates)) if best_i is None else int(best_i)
        global_round_picks += 1
        if local_div_after_first and remaining:
            v = embs[pick]
            sims = embs @ v
            r_idx = np.fromiter(remaining, dtype=int)
            global_round_max_sim[r_idx] = np.maximum(global_round_max_sim[r_idx], sims[r_idx])
        return pick

    def _pick_reveal_pred_deficit_h() -> Optional[int]:
        deficits = {
            str(g): max(0, int(k) - int(accepted_counts.get(g, 0)))
            for g in target_labels
        }
        deficit_labels = [
            g
            for g in sorted(
                deficits.keys(),
                key=lambda g: (-int(deficits.get(g, 0)), _gold_sort_key(str(g))),
            )
            if int(deficits.get(g, 0)) > 0
        ]
        if not deficit_labels:
            return None
        for def_label in deficit_labels:
            target_canon = _canon_label(def_label)
            if target_canon is None:
                continue
            bucket = [i for i in remaining if _canon_label(pred_labels[i]) == target_canon]
            if not bucket:
                continue
            best_i: Optional[int] = None
            best_h = -1.0
            for i in bucket:
                h = float(base["_H"].iat[i])
                if (h > best_h) or (np.isclose(h, best_h) and (best_i is None or i < best_i)):
                    best_i = i
                    best_h = h
            if best_i is not None:
                return int(best_i)
        return None

    def _pick_reveal_component(component: str) -> Optional[int]:
        if component not in {"uncertainty", "diversity"}:
            raise ValueError("component must be one of {'uncertainty','diversity'}")
        candidates = list(remaining)
        if not candidates:
            return None
        best_i: Optional[int] = None
        best_primary = -1.0
        best_secondary = -1.0
        for i in candidates:
            h = float(base["_H"].iat[i])
            d = _reveal_diversity(i)
            primary = h if component == "uncertainty" else d
            secondary = d if component == "uncertainty" else h
            if (
                (primary > best_primary)
                or (np.isclose(primary, best_primary) and secondary > best_secondary)
                or (
                    np.isclose(primary, best_primary)
                    and np.isclose(secondary, best_secondary)
                    and (best_i is None or i < best_i)
                )
            ):
                best_i = i
                best_primary = primary
                best_secondary = secondary
        return None if best_i is None else int(best_i)

    def _pick_reveal_pred_round_robin() -> Optional[int]:
        nonlocal reveal_pred_pointer
        if not pred_order:
            return None

        bucket: List[int] = []
        for _ in range(len(pred_order)):
            p = pred_order[reveal_pred_pointer]
            reveal_pred_pointer = (reveal_pred_pointer + 1) % len(pred_order)
            cand = sorted(i for i in remaining if pred_labels[i] == p)
            if cand:
                bucket = cand
                break

        if not bucket:
            return None
        # No scoring for this regime: take the next available item in the
        # round-robin predicted bucket.
        return int(bucket[0])

    def _pick_reveal_pred_round_robin_uncertainty_weighted() -> Optional[int]:
        nonlocal reveal_pred_pointer
        if not pred_order:
            return None

        bucket: List[int] = []
        for _ in range(len(pred_order)):
            p = pred_order[reveal_pred_pointer]
            reveal_pred_pointer = (reveal_pred_pointer + 1) % len(pred_order)
            cand = [i for i in remaining if pred_labels[i] == p]
            if cand:
                bucket = cand
                break

        if not bucket:
            return None

        weights = np.asarray(
            [max(0.0, float(base[entropy_col].iat[i])) for i in bucket],
            dtype=float,
        )
        total = float(np.sum(weights))
        if not np.isfinite(total) or total <= 0.0:
            # If all entropies are zero/invalid, fallback to uniform sampling.
            return int(rng.choice(bucket))
        probs = weights / total
        return int(rng.choice(bucket, p=probs))

    def _harmonic3(a: float, b: float, c: float) -> float:
        vals = [float(a), float(b), float(c)]
        if any(v <= 0.0 for v in vals):
            return 0.0
        return 3.0 / sum(1.0 / v for v in vals)

    def _counts_prior_from_obj(obj: Any) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if isinstance(obj, dict):
            for key, val in obj.items():
                canon = _canon_label(key)
                if canon is None:
                    continue
                try:
                    cnt = max(0.0, float(val))
                except Exception:
                    cnt = 0.0
                out[canon] = out.get(canon, 0.0) + cnt
        return out

    def _counts_to_granule_key(obj: Any) -> str:
        if not isinstance(obj, dict):
            return "__NO_COUNTS__"
        pairs: List[Tuple[str, int]] = []
        for key, val in obj.items():
            key_s = str(key)
            try:
                val_i = int(val)
            except Exception:
                try:
                    val_i = int(float(val))
                except Exception:
                    val_i = 0
            pairs.append((key_s, val_i))
        pairs.sort(key=lambda x: x[0])
        return str(pairs)

    granule_prior_round_robin_enabled = bool(granule_prior_round_robin)
    class_cycle: List[str] = []
    class_canons: List[str] = []
    granule_keys: List[str] = []
    granule_to_all_idxs: Dict[str, List[int]] = {}
    remaining_by_granule: Dict[str, set[int]] = {}
    granule_prior: Dict[str, Dict[str, float]] = {}
    granule_entropy: Dict[str, float] = {}
    granule_avg_sim: Dict[str, float] = {}
    granule_rr_pointer = 0
    granule_prior_cap_clamped = float(min(max(granule_prior_cap, 0.0), 1.0))

    if granule_prior_round_robin_enabled:
        class_cycle = [g for g in sorted(target_labels, key=_gold_sort_key) if g is not None]
        for g in class_cycle:
            canon = _canon_label(g)
            if canon is None:
                canon = str(g)
            if canon not in class_canons:
                class_canons.append(canon)

        if not class_cycle or not class_canons:
            granule_prior_round_robin_enabled = False
        else:
            base = base.copy()
            if "prediction_vector" in base.columns:
                base["_granule_key"] = (
                    base["prediction_vector"].fillna("__MISSING_VECTOR__").astype(str)
                )
            elif "counts" in base.columns:
                base["_granule_key"] = base["counts"].map(_counts_to_granule_key)
            else:
                base["_granule_key"] = base[pred_col].astype(str)

            granule_keys = base["_granule_key"].astype(str).tolist()
            for i, gk in enumerate(granule_keys):
                granule_to_all_idxs.setdefault(gk, []).append(i)
            remaining_by_granule = {
                gk: set(idxs) for gk, idxs in granule_to_all_idxs.items()
            }

            for gk, idxs in granule_to_all_idxs.items():
                idx0 = idxs[0]
                prior_counts: Dict[str, float] = {}
                if "prediction_vector" in base.columns:
                    pv = str(base["prediction_vector"].iat[idx0])
                    for tok in pv.split("|"):
                        canon = _canon_label(tok)
                        if canon is None:
                            continue
                        prior_counts[canon] = prior_counts.get(canon, 0.0) + 1.0
                if not prior_counts and "counts" in base.columns:
                    prior_counts = _counts_prior_from_obj(base["counts"].iat[idx0])
                if not prior_counts:
                    canon = _canon_label(base[pred_col].iat[idx0])
                    if canon is not None:
                        prior_counts[canon] = 1.0

                prior = {c: float(prior_counts.get(c, 0.0)) for c in class_canons}
                total_prior = float(sum(prior.values()))
                if total_prior <= 0.0:
                    uniform = 1.0 / float(len(class_canons))
                    prior = {c: uniform for c in class_canons}
                else:
                    prior = {c: (v / total_prior) for c, v in prior.items()}
                granule_prior[gk] = prior

                ent_vals = base.loc[idxs, entropy_col].astype(float).to_numpy()
                granule_entropy[gk] = float(np.mean(ent_vals)) if ent_vals.size > 0 else 0.0

                if len(idxs) <= 1:
                    granule_avg_sim[gk] = 1.0
                else:
                    arr = np.asarray(idxs, dtype=int)
                    ge = embs[arr]
                    sims = ge @ ge.T
                    tri = sims[np.triu_indices(len(arr), k=1)]
                    granule_avg_sim[gk] = float(np.mean(tri)) if tri.size > 0 else 1.0

    def _target_class_index(label: Optional[str]) -> Optional[int]:
        if label is None:
            return None
        try:
            return target_labels.index(str(label))
        except ValueError:
            pass
        canon = _canon_label(label)
        for idx, g in enumerate(target_labels):
            if _canon_label(g) == canon:
                return idx
        return None

    prob_rr_sampling_enabled = bool(
        (
            reveal_ping_pred_round_robin_prob_update
            or reveal_ping_pred_round_robin_prob_no_update
        )
        and reveal_one_by_one
        and len(target_labels) > 0
    )
    prob_update_rr_pointer = 0
    prob_update_class_probs: Optional[np.ndarray] = None
    prob_update_sim_matrix: Optional[np.ndarray] = None
    prob_update_sim_mean = 0.0

    if prob_rr_sampling_enabled:
        num_classes = len(target_labels)
        prob_update_class_probs = np.zeros((pool_size, num_classes), dtype=np.float64)
        counts_values = (
            base["counts"].tolist()
            if "counts" in base.columns
            else [None for _ in range(pool_size)]
        )
        uniform_row = np.full(num_classes, 1.0 / float(num_classes), dtype=np.float64)

        for i in range(pool_size):
            row = prob_update_class_probs[i]
            counts_obj = counts_values[i]
            if isinstance(counts_obj, dict):
                for key, val in counts_obj.items():
                    idx = _target_class_index(str(key))
                    if idx is None:
                        continue
                    try:
                        w = max(0.0, float(val))
                    except Exception:
                        w = 0.0
                    row[idx] += w

            if float(np.sum(row)) <= 0.0:
                idx = _target_class_index(pred_labels[i])
                if idx is not None:
                    row[idx] = 1.0

            row_sum = float(np.sum(row))
            if row_sum > 0.0 and np.isfinite(row_sum):
                prob_update_class_probs[i] = row / row_sum
            else:
                prob_update_class_probs[i] = uniform_row

        if reveal_ping_pred_round_robin_prob_update:
            prob_update_sim_matrix = (embs @ embs.T).astype(np.float32, copy=False)
            if pool_size > 1:
                tri = prob_update_sim_matrix[np.triu_indices(pool_size, k=1)]
                prob_update_sim_mean = float(np.mean(tri)) if tri.size > 0 else 0.0
            else:
                prob_update_sim_mean = 0.0

    while remaining and not _quotas_satisfied():
        mode = "ping"
        expected_target_idx: Optional[int] = None
        if granule_prior_round_robin_enabled:
            active_classes = [g for g in class_cycle if int(accepted_counts.get(g, 0)) < int(k)]
            if not active_classes:
                break

            target_class: Optional[str] = None
            for _ in range(len(class_cycle)):
                cand = class_cycle[granule_rr_pointer]
                granule_rr_pointer = (granule_rr_pointer + 1) % len(class_cycle)
                if cand in active_classes:
                    target_class = cand
                    break
            if target_class is None:
                break
            target_canon = _canon_label(target_class)
            if target_canon is None:
                target_canon = str(target_class)

            pick_maybe: Optional[int] = None
            available_granules = [gk for gk, idxs in remaining_by_granule.items() if idxs]
            mode = "ping_granule_prior_rr"
            if available_granules:
                non_singletons = [
                    gk for gk in available_granules if len(remaining_by_granule[gk]) > 1
                ]
                candidate_granules = non_singletons if non_singletons else available_granules
                max_size = (
                    max(len(remaining_by_granule[gk]) for gk in candidate_granules)
                    if candidate_granules
                    else 0
                )
                granule_scores: List[float] = []
                for gk in candidate_granules:
                    p_t = float(granule_prior.get(gk, {}).get(target_canon, 0.0))
                    h_g = float(granule_entropy.get(gk, 0.0))
                    size_norm = (
                        float(len(remaining_by_granule[gk])) / float(max_size)
                        if max_size > 0
                        else 0.0
                    )
                    s_g = _harmonic3(p_t, h_g, size_norm)
                    granule_scores.append(max(0.0, float(s_g)))

                if candidate_granules:
                    probs = np.asarray(granule_scores, dtype=float)
                    total = float(np.sum(probs))
                    if np.isfinite(total) and total > 0.0:
                        probs = probs / total
                    else:
                        probs = np.full(
                            len(candidate_granules),
                            1.0 / float(len(candidate_granules)),
                            dtype=float,
                        )
                    chosen_granule = str(rng.choice(candidate_granules, p=probs))
                    granule_pool = sorted(remaining_by_granule[chosen_granule])
                    if granule_pool:
                        pick_maybe = int(rng.choice(granule_pool))

            if pick_maybe is None:
                remain_sorted = sorted(int(i) for i in remaining)
                if not remain_sorted:
                    break
                pick = int(rng.choice(remain_sorted))
                mode = "ping_granule_fallback_random"
            else:
                pick = int(pick_maybe)
        elif pure_random:
            pick = int(rng.choice(list(remaining)))
            mode = "random_pool"
        elif reveal_one_by_one:
            if (
                (reveal_ping_pred_round_robin_prob_update or reveal_ping_pred_round_robin_prob_no_update)
                and prob_rr_sampling_enabled
            ):
                active_classes = [g for g in target_labels if int(accepted_counts.get(g, 0)) < int(k)]
                if not active_classes:
                    break
                target_label: Optional[str] = None
                for _ in range(len(target_labels)):
                    cand = target_labels[prob_update_rr_pointer]
                    prob_update_rr_pointer = (prob_update_rr_pointer + 1) % len(target_labels)
                    if int(accepted_counts.get(cand, 0)) < int(k):
                        target_label = cand
                        break
                if target_label is None:
                    break

                target_idx = _target_class_index(target_label)
                r_idx = np.fromiter(remaining, dtype=int)
                if target_idx is None or r_idx.size == 0:
                    if r_idx.size == 0:
                        break
                    pick = int(rng.choice(r_idx))
                else:
                    weights = np.asarray(prob_update_class_probs[r_idx, target_idx], dtype=np.float64)
                    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
                    total = float(np.sum(weights))
                    if total > 0.0:
                        probs = weights / total
                        pick = int(rng.choice(r_idx, p=probs))
                    else:
                        pick = int(rng.choice(r_idx))
                    expected_target_idx = int(target_idx)
                mode = (
                    "ping_reveal_pred_round_robin_prob_update"
                    if reveal_ping_pred_round_robin_prob_update
                    else "ping_reveal_pred_round_robin_prob_no_update"
                )
            elif reveal_ping_pred_round_robin_uncertainty_weighted:
                pick_maybe = _pick_reveal_pred_round_robin_uncertainty_weighted()
                if pick_maybe is None:
                    remain_sorted = sorted(int(i) for i in remaining)
                    pick_maybe = remain_sorted[0] if remain_sorted else None
                if pick_maybe is None:
                    break
                pick = int(pick_maybe)
                mode = "ping_reveal_pred_round_robin_uncertainty_weighted"
            elif reveal_ping_pred_round_robin:
                pick_maybe = _pick_reveal_pred_round_robin()
                if pick_maybe is None:
                    remain_sorted = sorted(int(i) for i in remaining)
                    pick_maybe = remain_sorted[0] if remain_sorted else None
                if pick_maybe is None:
                    break
                pick = int(pick_maybe)
                mode = "ping_reveal_pred_round_robin"
            elif reveal_ping_pred_deficit_h:
                pick_maybe = _pick_reveal_pred_deficit_h()
                if pick_maybe is None:
                    pick_maybe = _best_by_score(
                        list(remaining), mode=score_mode, reveal_div_mode=True
                    )
                pick = int(rng.choice(list(remaining))) if pick_maybe is None else int(pick_maybe)
                mode = "ping_reveal_pred_deficit_h"
            elif reveal_ping_random_pool:
                pick = int(rng.choice(list(remaining)))
                mode = "ping_reveal_random_pool"
            elif reveal_ping_alt_ud:
                component = "uncertainty" if (reveal_component_count % 2 == 0) else "diversity"
                pick_maybe = _pick_reveal_component(component)
                if pick_maybe is None:
                    pick_maybe = _best_by_score(
                        list(remaining), mode=score_mode, reveal_div_mode=True
                    )
                pick = int(rng.choice(list(remaining))) if pick_maybe is None else int(pick_maybe)
                mode = "ping_reveal_uncertainty" if component == "uncertainty" else "ping_reveal_diversity"
                reveal_component_count += 1
            else:
                pick_maybe = _best_by_score(
                    list(remaining), mode=score_mode, reveal_div_mode=True
                )
                pick = int(rng.choice(list(remaining))) if pick_maybe is None else int(pick_maybe)
                mode = "ping_reveal"
        else:
            if use_pred_round_robin:
                pick = _pick_pred_round_robin()
                mode = "ping_round_robin"
            else:
                pick = _pick_global_pool()
                mode = "ping_global"

        if pick not in remaining:
            break
        remaining.remove(pick)
        annotated_order.append(pick)
        if not granule_prior_round_robin_enabled:
            _update_similarity_after_pick(pick)

        g = gold_labels[pick]
        if (
            reveal_ping_pred_round_robin_prob_update
            and prob_rr_sampling_enabled
            and expected_target_idx is not None
            and prob_update_class_probs is not None
            and prob_update_sim_matrix is not None
        ):
            revealed_idx = _target_class_index(g)
            if revealed_idx is not None and int(revealed_idx) != int(expected_target_idx):
                if remaining:
                    r_idx = np.fromiter(remaining, dtype=int)
                    sims = prob_update_sim_matrix[pick, r_idx]
                    affected = r_idx[sims > float(prob_update_sim_mean)]
                    if affected.size > 0:
                        moved = prob_update_class_probs[affected, expected_target_idx].copy()
                        prob_update_class_probs[affected, expected_target_idx] = 0.0
                        prob_update_class_probs[affected, revealed_idx] += moved

        if granule_prior_round_robin_enabled:
            gk_pick = granule_keys[pick] if pick < len(granule_keys) else None
            if gk_pick is not None and gk_pick in remaining_by_granule:
                remaining_by_granule[gk_pick].discard(pick)

            g_canon = _canon_label(g) if g is not None else None
            if (
                g_canon is not None
                and gk_pick is not None
                and gk_pick in granule_prior
                and g_canon in granule_prior[gk_pick]
            ):
                all_idx_arr = np.asarray(granule_to_all_idxs[gk_pick], dtype=int)
                if all_idx_arr.size > 0:
                    sims_to_pick = embs[all_idx_arr] @ embs[pick]
                    frac = float(
                        np.mean(sims_to_pick > float(granule_avg_sim.get(gk_pick, 1.0)))
                    )
                else:
                    frac = 0.0

                p = granule_prior[gk_pick]
                old_true = float(p.get(g_canon, 0.0))
                new_true = min(granule_prior_cap_clamped, old_true + frac)
                rest = max(0.0, 1.0 - new_true)
                others = [c for c in p.keys() if c != g_canon]
                old_other_sum = float(sum(max(0.0, float(p.get(c, 0.0))) for c in others))
                if others:
                    if old_other_sum > 0.0:
                        for c in others:
                            p[c] = (max(0.0, float(p.get(c, 0.0))) / old_other_sum) * rest
                    else:
                        u = rest / float(len(others))
                        for c in others:
                            p[c] = u
                p[g_canon] = new_true
                p_total = float(sum(max(0.0, float(v)) for v in p.values()))
                if p_total > 0.0:
                    for c in list(p.keys()):
                        p[c] = max(0.0, float(p[c])) / p_total

        accepted = False
        if g is not None and g in observed_counts:
            observed_counts[g] += 1
        if g is not None and g in accepted_counts and accepted_counts[g] < int(k):
            accepted_counts[g] += 1
            accepted_order.append(pick)
            accepted = True

        if not accepted:
            pass

    annotated_total = int(len(annotated_order))
    selected_exemplars = int(len(accepted_order))
    useless_annotated = int(max(0, annotated_total - selected_exemplars))
    target_exemplars = int(k) * int(len(target_labels))
    completed = bool(_quotas_satisfied())
    ratio = (float(selected_exemplars) / float(annotated_total)) if annotated_total > 0 else 0.0
    completion_fraction = (
        float(selected_exemplars) / float(target_exemplars)
        if target_exemplars > 0
        else 0.0
    )

    return {
        "annotated_total": annotated_total,
        "selected_exemplars": selected_exemplars,
        "useless_annotated": useless_annotated,
        "selected_over_annotated_ratio": ratio,
        "completion_fraction": completion_fraction,
        "completed": completed,
        "target_exemplars": target_exemplars,
        "num_target_classes": int(len(target_labels)),
        "pool_size": int(pool_size),
        "possible_to_complete": bool(possible_to_complete),
        "accepted_counts_json": json.dumps(accepted_counts, sort_keys=True),
        "observed_counts_json": json.dumps(observed_counts, sort_keys=True),
    }


def compute_balanced_draw_stats_for_regime(
    regime_name: str,
    regime_version: int,
    regime_kwargs: Dict[str, Any],
    *,
    seed_k_pairs: Sequence[Tuple[int, int]],
    models: Sequence[str],
    doc_ids: Sequence[str],
    base_predictions: pd.DataFrame,
    gold_label_for_id: Any,
    shared_resources: Any,
    gold_df: pd.DataFrame,
    embed_model: str,
    embed_device: str,
    embed_prefix: str,
    on_pair_complete: Optional[Callable[[Tuple[int, int], pd.DataFrame], None]] = None,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    committee_cache = {}
    for target_model in models:
        committee_models = [m for m in models if m != target_model]
        committee_preds = base_predictions[base_predictions["model"].isin(committee_models)]
        committee_cache[target_model] = prepare_loo_resources(
            committee_preds,
            docs=None,
            text_lookup=shared_resources.text_lookup,
            include_models=committee_models,
            gold_df=gold_df,
            existing_embedding_cache=shared_resources.embedding_cache,
            compute_embeddings=False,
            embed_model=embed_model,
            embed_device=embed_device,
            embed_prefix=embed_prefix,
        )

    by_seed: Dict[int, List[int]] = {}
    for s, k in seed_k_pairs:
        by_seed.setdefault(int(s), []).append(int(k))

    for seed in sorted(by_seed.keys()):
        ks = sorted(set(by_seed[seed]))
        for k in ks:
            pair_start_idx = len(rows)
            pbar = tqdm(models, desc=f"{regime_name} | seed={seed} | k={k}", leave=False)
            for target_model in pbar:
                res = committee_cache[target_model]
                table = res.vote_table.copy()
                id_col, pred_col, ent_col = res.id_col, res.pred_col, res.entropy_col

                table[id_col] = table[id_col].astype(str)
                table["_gold"] = table[id_col].map(gold_label_for_id)

                table_by_id = table.set_index(id_col, drop=False)
                target_ids = [fid for fid in doc_ids if fid in table_by_id.index]

                for target_doc in target_ids:
                    pool = table_by_id.drop(target_doc).reset_index(drop=True)
                    if pool.empty:
                        continue

                    summary = simulate_balanced_draw_no_comp(
                        pool,
                        k=int(k),
                        id_col=id_col,
                        pred_col=pred_col,
                        entropy_col=ent_col,
                        gold_col="_gold",
                        embedding_cache=res.embedding_cache,
                        seed=int(seed),
                        regime_kwargs=regime_kwargs,
                    )

                    rows.append(
                        {
                            "regime": regime_name,
                            "regime_version": int(regime_version),
                            "seed": int(seed),
                            "k": int(k),
                            "target_model": str(target_model),
                            "target_doc": str(target_doc),
                            **summary,
                            "score": str(regime_kwargs.get("score", "harmonic")),
                            "use_pred_round_robin": bool(regime_kwargs.get("use_pred_round_robin", True)),
                            "global_pool_round_local_diversity_after_first": bool(
                                regime_kwargs.get(
                                    "global_pool_round_local_diversity_after_first",
                                    False,
                                )
                            ),
                            "reveal_one_by_one": bool(regime_kwargs.get("reveal_one_by_one", False)),
                            "reveal_ping_pred_round_robin": bool(
                                regime_kwargs.get("reveal_ping_pred_round_robin", False)
                            ),
                            "reveal_ping_pred_round_robin_uncertainty_weighted": bool(
                                regime_kwargs.get(
                                    "reveal_ping_pred_round_robin_uncertainty_weighted",
                                    False,
                                )
                            ),
                            "reveal_ping_pred_round_robin_prob_update": bool(
                                regime_kwargs.get(
                                    "reveal_ping_pred_round_robin_prob_update",
                                    False,
                                )
                            ),
                            "reveal_ping_pred_round_robin_prob_no_update": bool(
                                regime_kwargs.get(
                                    "reveal_ping_pred_round_robin_prob_no_update",
                                    False,
                                )
                            ),
                            "reveal_ping_pred_deficit_h": bool(
                                regime_kwargs.get("reveal_ping_pred_deficit_h", False)
                            ),
                            "reveal_ping_random_pool": bool(
                                regime_kwargs.get("reveal_ping_random_pool", False)
                            ),
                            "reveal_ping_alternate_uncertainty_diversity": bool(
                                regime_kwargs.get(
                                    "reveal_ping_alternate_uncertainty_diversity",
                                    False,
                                )
                            ),
                            "granule_prior_round_robin": bool(
                                regime_kwargs.get("granule_prior_round_robin", False)
                            ),
                            "granule_prior_cap": float(
                                regime_kwargs.get("granule_prior_cap", 0.95)
                            ),
                            "pure_random": bool(regime_kwargs.get("pure_random", False)),
                        }
                    )

            if on_pair_complete is not None:
                pair_rows = rows[pair_start_idx:]
                pair_df = pd.DataFrame(pair_rows) if pair_rows else pd.DataFrame()
                on_pair_complete((int(seed), int(k)), pair_df)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def aggregate_balanced_draw_stats(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame()

    work = df.copy()
    for col in [
        "annotated_total",
        "selected_exemplars",
        "useless_annotated",
        "target_exemplars",
        "selected_over_annotated_ratio",
        "completion_fraction",
        "completed",
    ]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    agg_seed_k = (
        work.groupby(["regime", "seed", "k"], as_index=False)
        .agg(
            annotated_total=("annotated_total", "sum"),
            selected_exemplars=("selected_exemplars", "sum"),
            useless_annotated=("useless_annotated", "sum"),
            target_exemplars=("target_exemplars", "sum"),
            mean_selected_over_annotated_ratio=("selected_over_annotated_ratio", "mean"),
            mean_completion_fraction=("completion_fraction", "mean"),
            completed_targets=("completed", "sum"),
            n_targets=("target_doc", "count"),
        )
        .sort_values(["regime", "seed", "k"])
        .reset_index(drop=True)
    )
    agg_seed_k["global_selected_over_annotated_ratio"] = (
        agg_seed_k["selected_exemplars"] / agg_seed_k["annotated_total"].replace(0, np.nan)
    )
    agg_seed_k["global_target_over_annotated_ratio"] = (
        agg_seed_k["target_exemplars"] / agg_seed_k["annotated_total"].replace(0, np.nan)
    )
    agg_seed_k["completed_rate"] = (
        agg_seed_k["completed_targets"] / agg_seed_k["n_targets"].replace(0, np.nan)
    )

    agg_k = (
        work.groupby(["regime", "k"], as_index=False)
        .agg(
            annotated_total=("annotated_total", "sum"),
            selected_exemplars=("selected_exemplars", "sum"),
            useless_annotated=("useless_annotated", "sum"),
            target_exemplars=("target_exemplars", "sum"),
            mean_selected_over_annotated_ratio=("selected_over_annotated_ratio", "mean"),
            mean_completion_fraction=("completion_fraction", "mean"),
            completed_targets=("completed", "sum"),
            n_targets=("target_doc", "count"),
            n_seeds=("seed", "nunique"),
        )
        .sort_values(["regime", "k"])
        .reset_index(drop=True)
    )
    agg_k["global_selected_over_annotated_ratio"] = (
        agg_k["selected_exemplars"] / agg_k["annotated_total"].replace(0, np.nan)
    )
    agg_k["global_target_over_annotated_ratio"] = (
        agg_k["target_exemplars"] / agg_k["annotated_total"].replace(0, np.nan)
    )
    agg_k["completed_rate"] = agg_k["completed_targets"] / agg_k["n_targets"].replace(0, np.nan)

    return agg_seed_k, agg_k


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Balanced draw simulation for ping/pong regimes without compensation turns.",
    )
    parser.add_argument("--config", type=str, default="configs/mimic_baseline_loo_config.json")
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--base-predictions", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--doc-limit", type=int, default=None)
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seeds (e.g. 0,300)")
    parser.add_argument("--k-values", type=str, default=None, help="Comma-separated k values (e.g. 2,3,4,5)")
    parser.add_argument("--model-subset", type=str, default=None, help="Comma-separated model names")
    parser.add_argument("--regimes", type=str, default=None, help="Comma-separated regime names")
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument(
        "--analysis-subdir",
        type=str,
        default="analysis/ping_pong_balanced_draw_stats",
        help="Relative path under output_root for cached CSVs.",
    )
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    cfg = json.loads(Path(args.config).read_text())
    loo_cfg = dict(cfg.get("loo", {}))
    models_cfg = list(cfg.get("models", []))

    dataset, data_dir, gold_path = _resolve_dataset_paths(cfg, args.dataset)
    output_root = _resolve_output_root(cfg, args.output_root)
    base_predictions_path = _resolve_base_predictions_path(cfg, output_root, args.base_predictions)
    out_dir = output_root / str(args.analysis_subdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    docs_all = load_documents(str(data_dir))
    if not docs_all:
        raise RuntimeError(f"No documents loaded from {data_dir}")
    docs = docs_all if args.doc_limit is None else docs_all[: int(args.doc_limit)]
    doc_ids = [str(d["file"]) for d in docs]
    doc_id_set = set(doc_ids)

    base_predictions = pd.read_pickle(base_predictions_path).copy()
    base_predictions["file"] = base_predictions["file"].astype(str)

    available_models = set(base_predictions["model"].unique())
    models = [m for m in models_cfg if m in available_models]
    model_subset = _parse_str_list(args.model_subset)
    if model_subset is not None:
        subset = set(model_subset)
        models = [m for m in models if m in subset]
    if not models:
        raise RuntimeError("No configured models found in base predictions.")

    model_filtered = base_predictions[base_predictions["model"].isin(models)].copy()
    overlap_files = set(model_filtered["file"].astype(str).unique()) & doc_id_set
    base_predictions = model_filtered[model_filtered["file"].isin(doc_id_set)].reset_index(drop=True)
    if base_predictions.empty:
        raise RuntimeError(
            "Base predictions are empty after filtering by docs/models. "
            f"dataset={dataset}, docs={len(doc_ids)}, model_count={len(models)}, file_overlap={len(overlap_files)}."
        )

    gold_df = pd.read_pickle(gold_path)
    if "file" not in gold_df.columns or "label" not in gold_df.columns:
        raise KeyError("Gold DataFrame must contain columns ['file', 'label'].")

    gold_keys = tuple(str(x) for x in gold_df["file"].tolist())
    gold_lookup = {str(k): str(v) for k, v in zip(gold_df["file"].tolist(), gold_df["label"].tolist())}

    @lru_cache(maxsize=None)
    def gold_label_for_id(doc_id: str) -> Optional[str]:
        s = str(doc_id)
        if s in gold_lookup:
            return gold_lookup[s]
        match = next((g for g in gold_keys if g in s), None)
        return gold_lookup.get(match) if match is not None else None

    seeds = _parse_int_list(args.seeds)
    if seeds is None:
        seeds = [int(s) for s in loo_cfg.get("seeds", [0])]

    k_values = _parse_int_list(args.k_values)
    if k_values is None:
        k_values = [int(k) for k in loo_cfg.get("k_values", [2, 4, 6, 8, 10])]

    embed_model = str(loo_cfg.get("embed_model", "intfloat/multilingual-e5-large"))
    embed_prefix = str(loo_cfg.get("embed_prefix", "passage: "))
    embed_device_cfg = str(loo_cfg.get("embed_device", "auto"))
    if embed_device_cfg == "auto":
        embed_device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
    else:
        embed_device = embed_device_cfg

    all_regimes = build_regime_specs_no_compensation()
    requested_regimes = _parse_str_list(args.regimes)
    if requested_regimes is None:
        regime_specs = all_regimes
    else:
        missing = [r for r in requested_regimes if r not in all_regimes]
        if missing:
            raise KeyError(f"Unknown regime(s): {missing}. Available: {sorted(all_regimes.keys())}")
        regime_specs = {name: all_regimes[name] for name in requested_regimes}

    print(f"project_root={PROJECT_ROOT}")
    print(f"dataset={dataset}")
    print(f"data_dir={data_dir}")
    print(f"gold_path={gold_path}")
    print(f"output_root={output_root}")
    print(f"base_predictions_path={base_predictions_path}")
    print(f"docs={len(docs)} / {len(docs_all)}")
    print(f"models={len(models)}")
    print(f"seeds={seeds}")
    print(f"k_values={k_values}")
    print(f"embed_device={embed_device}")
    print(f"regimes={list(regime_specs.keys())}")

    shared_resources = prepare_loo_resources(
        base_predictions,
        docs=docs,
        include_models=models,
        gold_df=gold_df,
        embed_model=embed_model,
        embed_device=embed_device,
        embed_prefix=embed_prefix,
    )
    print("shared vote rows:", len(shared_resources.vote_table))
    print(
        "embedding cache size:",
        0 if shared_resources.embedding_cache is None else len(shared_resources.embedding_cache),
    )

    cache_stats_path = out_dir / "balanced_draw_stats_per_target.csv"
    agg_seed_k_path = out_dir / "balanced_draw_stats_agg_seed_k.csv"
    agg_k_path = out_dir / "balanced_draw_stats_agg_k.csv"

    cached_df = pd.read_csv(cache_stats_path) if cache_stats_path.exists() else pd.DataFrame()
    if not cached_df.empty and "regime_version" not in cached_df.columns:
        cached_df = cached_df.copy()
        cached_df["regime_version"] = 1

    reuse_parts: List[pd.DataFrame] = []
    recompute_parts: List[pd.DataFrame] = []
    requested = set(regime_specs.keys())

    if not cached_df.empty:
        other_cached = cached_df[~cached_df["regime"].isin(requested)].copy()
    else:
        other_cached = pd.DataFrame()

    def _compose_stats_df() -> pd.DataFrame:
        parts = []
        if not other_cached.empty:
            parts.append(other_cached)
        parts.extend(reuse_parts)
        parts.extend(recompute_parts)
        if not parts:
            return pd.DataFrame()
        df_now = pd.concat(parts, ignore_index=True)
        df_now = df_now.sort_values(
            ["regime", "seed", "k", "target_model", "target_doc"]
        ).reset_index(drop=True)
        return df_now

    def _write_checkpoint(tag: str) -> None:
        stats_now = _compose_stats_df()
        if stats_now.empty:
            return
        stats_now.to_csv(cache_stats_path, index=False)
        agg_seed_k_now, agg_k_now = aggregate_balanced_draw_stats(stats_now)
        agg_seed_k_now.to_csv(agg_seed_k_path, index=False)
        agg_k_now.to_csv(agg_k_path, index=False)
        print(
            f"[checkpoint] {tag}: per-target={len(stats_now)} "
            f"agg_seed_k={len(agg_seed_k_now)} agg_k={len(agg_k_now)}"
        )

    for regime_name, spec in regime_specs.items():
        version = int(spec["version"])
        kwargs = dict(spec["kwargs"])

        cached_regime = pd.DataFrame()
        if not cached_df.empty:
            cached_regime = cached_df[
                (cached_df["regime"] == regime_name)
                & (cached_df["regime_version"].astype(int) == version)
            ].copy()

        cached_pairs = set()
        if not cached_regime.empty:
            cached_pairs = {
                (int(r.seed), int(r.k))
                for r in cached_regime[["seed", "k"]].drop_duplicates().itertuples(index=False)
            }

        expected_pairs = _seed_k_pairs_for_regime(
            regime_name,
            seeds=seeds,
            k_values=k_values,
        )
        effective_seeds = sorted({int(s) for s, _ in expected_pairs})
        if len(effective_seeds) != len(seeds):
            print(
                f"[info] {regime_name}: deterministic regime; using seeds={effective_seeds} "
                f"(from requested {seeds})"
            )

        if args.force_recompute:
            missing_pairs = set(expected_pairs)
            keep_pairs: set[Tuple[int, int]] = set()
        else:
            missing_pairs = expected_pairs - cached_pairs
            keep_pairs = expected_pairs & cached_pairs

        if keep_pairs and not cached_regime.empty:
            reuse_parts.append(_filter_seed_k_pairs(cached_regime, keep_pairs))

        if missing_pairs:
            print(
                f"[compute] {regime_name}: computing {len(missing_pairs)} seed/k pairs "
                f"(version={version})"
            )

            def _on_pair_complete(pair: Tuple[int, int], pair_df: pd.DataFrame) -> None:
                pair_seed, pair_k = int(pair[0]), int(pair[1])
                if pair_df is None or pair_df.empty:
                    print(
                        f"[checkpoint] {regime_name} seed={pair_seed} k={pair_k}: "
                        "no rows produced."
                    )
                    return
                recompute_parts.append(pair_df)
                _write_checkpoint(f"{regime_name} seed={pair_seed} k={pair_k}")

            _ = compute_balanced_draw_stats_for_regime(
                regime_name,
                version,
                kwargs,
                seed_k_pairs=sorted(missing_pairs),
                models=models,
                doc_ids=doc_ids,
                base_predictions=base_predictions,
                gold_label_for_id=gold_label_for_id,
                shared_resources=shared_resources,
                gold_df=gold_df,
                embed_model=embed_model,
                embed_device=embed_device,
                embed_prefix=embed_prefix,
                on_pair_complete=_on_pair_complete,
            )
        else:
            print(f"[cache] {regime_name}: reusing cached rows (version={version})")

    stats_df = _compose_stats_df()
    if stats_df.empty:
        raise RuntimeError("No stats produced.")

    stats_df.to_csv(cache_stats_path, index=False)

    agg_seed_k_df, agg_k_df = aggregate_balanced_draw_stats(stats_df)
    agg_seed_k_df.to_csv(agg_seed_k_path, index=False)
    agg_k_df.to_csv(agg_k_path, index=False)

    print(f"wrote per-target stats: {cache_stats_path} ({len(stats_df)} rows)")
    print(f"wrote agg seed/k stats: {agg_seed_k_path} ({len(agg_seed_k_df)} rows)")
    print(f"wrote agg k stats: {agg_k_path} ({len(agg_k_df)} rows)")


if __name__ == "__main__":
    main()
