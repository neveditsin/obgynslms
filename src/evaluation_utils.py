"""Utility helpers that were previously embedded in notebooks."""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Optional, Sequence, Tuple

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from .label_utils import canonicalize_label, infer_label

__all__ = [
    "attach_gold_and_preds",
    "collapse_predictions_by_document",
    "evaluate_by_model",
    "majority_vote_with_seed_priority",
]


def attach_gold_and_preds(
    results_df: pd.DataFrame,
    gold_df: pd.DataFrame,
    *,
    pred_col: str = "raw_output",
    file_col: str = "file",
    on: str = "file",
    case_insensitive: bool = True,
    prefer_longest_match: bool = True,
    join_how: str = "inner",
) -> pd.DataFrame:
    """Merge predictions with gold labels using substring matching."""

    if results_df is None or results_df.empty:
        return pd.DataFrame()

    df = results_df.copy()
    df["pred"] = df[pred_col].apply(infer_label)

    gold_norm = gold_df.copy()
    if "label" in gold_norm.columns:
        def _normalize_gold(value: object) -> object:
            canon = canonicalize_label(value)
            if canon is not None:
                return canon
            if pd.isna(value):
                return value
            return str(value).strip().lower()

        gold_norm["label"] = gold_norm["label"].map(_normalize_gold)

    gold_keys_orig = [str(x) for x in gold_norm[on].dropna().tolist()]
    gold_keys_norm = [g.lower() for g in gold_keys_orig] if case_insensitive else gold_keys_orig

    @lru_cache(maxsize=None)
    def _match_gold_key(file_value: Optional[str]) -> Optional[str]:
        if file_value is None:
            return None
        s = str(file_value)
        s_norm = s.lower() if case_insensitive else s

        best_key: Optional[str] = None
        best_len = -1
        for orig, norm in zip(gold_keys_orig, gold_keys_norm):
            if norm and norm in s_norm:
                if not prefer_longest_match:
                    return orig
                if len(norm) > best_len:
                    best_key = orig
                    best_len = len(norm)
        return best_key

    df["gold_file"] = df[file_col].map(_match_gold_key)

    merged = df.merge(
        gold_norm,
        left_on="gold_file",
        right_on=on,
        how=join_how,
        suffixes=("_pred", ""),
    )
    return merged


def majority_vote_with_seed_priority(
    preds: pd.Series,
    seeds: Optional[pd.Series] = None,
) -> Optional[str]:
    """Resolve a majority vote, breaking ties by deterministic seed priority."""

    if preds is None or preds.empty:
        return None

    counts = preds.value_counts(dropna=True)
    if counts.empty:
        return None

    top_count = counts.max()
    winners = counts[counts == top_count].index.tolist()

    if len(winners) == 1:
        return winners[0]

    if seeds is not None and not seeds.isna().all():
        priority = pd.DataFrame({"pred": preds, "seed": seeds})
        priority = priority.dropna(subset=["seed"]).sort_values(["seed", "pred"])
        for _, row in priority.iterrows():
            if row["pred"] in winners:
                return row["pred"]

    # deterministic fallback: alphabetical order
    return sorted(winners)[0]


def collapse_predictions_by_document(
    df: pd.DataFrame,
    *,
    doc_col: str = "file",
    pred_col: str = "pred",
    label_col: str = "label",
    seed_col: str = "seed",
    group_cols: Optional[Sequence[str]] = None,
    keep_first_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Collapse duplicate document predictions via majority vote with seed priority."""

    if df is None or df.empty:
        return pd.DataFrame(columns=[doc_col, pred_col, label_col])

    missing = [col for col in (doc_col, pred_col, label_col) if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for collapse: {missing}")

    if group_cols is None:
        group_cols = ["model", doc_col] if "model" in df.columns else [doc_col]
    else:
        group_cols = [col for col in group_cols if col in df.columns]
        if not group_cols:
            group_cols = ["model", doc_col] if "model" in df.columns else [doc_col]

    if doc_col not in group_cols:
        group_cols = list(group_cols) + [doc_col]

    keep_first_cols = [col for col in (keep_first_cols or []) if col in df.columns]

    def _collapse(group: pd.DataFrame) -> pd.Series:
        vote = majority_vote_with_seed_priority(
            group[pred_col], group[seed_col] if seed_col in group.columns else None
        )
        row = {
            label_col: group[label_col].iloc[0],
            pred_col: vote,
        }
        for col in keep_first_cols:
            row[col] = group[col].iloc[0]
        return pd.Series(row)

    collapsed = (
        df.groupby(group_cols, dropna=False)
        .apply(_collapse)
        .reset_index()
    )
    return collapsed


def evaluate_by_model(
    results_df: pd.DataFrame,
    gold_df: pd.DataFrame,
    *,
    group_col: str = "model",
    pred_col: str = "raw_output",
    on: str = "file",
    labels: Iterable[str] = ("early", "late", "unrelated"),
    collapse_duplicates: bool = False,
    collapse_group_cols: Optional[Sequence[str]] = None,
    collapse_keep_cols: Optional[Sequence[str]] = None,
    doc_col: str = "file",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute precision/recall/F1 grouped by ``group_col``."""

    merged = attach_gold_and_preds(
        results_df,
        gold_df,
        pred_col=pred_col,
        file_col="file",
        on=on,
        case_insensitive=True,
        prefer_longest_match=True,
        join_how="inner",
    )

    eval_frame = merged
    if collapse_duplicates:
        eval_frame = collapse_predictions_by_document(
            merged,
            doc_col=doc_col,
            pred_col="pred",
            label_col="label",
            seed_col="seed",
            group_cols=collapse_group_cols,
            keep_first_cols=collapse_keep_cols,
        )

    rows = []
    for name, group in eval_frame.groupby(group_col):
        p, r, f1, _ = precision_recall_fscore_support(
            group["label"],
            group["pred"],
            labels=list(labels),
            average="macro",
            zero_division=0,
        )
        rows.append({group_col: name, "precision": p, "recall": r, "f1": f1, "n": len(group)})

    summary = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
    return summary, eval_frame
