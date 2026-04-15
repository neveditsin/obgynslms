import numpy as np
import pandas as pd
from itertools import combinations
from typing import Callable, Dict, Tuple, Iterable, Optional
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

import numpy as np
import pandas as pd
from itertools import combinations
from typing import Callable, Dict, Tuple, Iterable, Optional, List
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

try:
    from tqdm import tqdm
except Exception:
    # fallback if tqdm not installed
    def tqdm(x, **kwargs):
        return x

# ----------------------------
# Metric plumbing (unchanged)
# ----------------------------
def get_metric_fn(metric: str = "f1", average: str = "macro", labels: Optional[Iterable[str]] = None) -> Callable:
    metric = metric.lower()
    if metric == "f1":
        return lambda y_true, y_pred: f1_score(y_true, y_pred, average=average, labels=list(labels) if labels else None, zero_division=0)
    if metric == "precision":
        return lambda y_true, y_pred: precision_score(y_true, y_pred, average=average, labels=list(labels) if labels else None, zero_division=0)
    if metric == "recall":
        return lambda y_true, y_pred: recall_score(y_true, y_pred, average=average, labels=list(labels) if labels else None, zero_division=0)
    if metric == "accuracy":
        return lambda y_true, y_pred: accuracy_score(y_true, y_pred)
    raise ValueError(f"Unsupported metric: {metric}")

# ----------------------------
# Helpers to make systems unique per file
# ----------------------------
def _majority_vote(series: pd.Series) -> str:
    counts = series.value_counts()
    # break ties deterministically by label sort
    top = counts[counts == counts.max()].index
    return sorted(top)[0]

def _dedupe_by_file(tbl: pd.DataFrame, agg_mode: str = "vote") -> pd.DataFrame:
    """
    Ensure a single (y_true, y_pred) per file.
    - agg_mode="vote": majority vote over y_pred, y_true taken as first (they should be identical per file).
    - agg_mode="first": take first row.
    """
    if agg_mode == "first":
        return tbl.sort_values("file").drop_duplicates("file", keep="first")

    # vote
    # y_true should be the same per file; take first
    y_true = tbl.groupby("file")["y_true"].first()
    y_pred = tbl.groupby("file")["y_pred"].apply(_majority_vote)
    out = pd.DataFrame({"file": y_true.index, "y_true": y_true.values})
    out["y_pred"] = y_pred.reindex(out["file"]).values
    return out.reset_index(drop=True)

# ----------------------------
# Core extraction (UPDATED)
# ----------------------------
def _extract_system_tables(
    dfs: Dict[str, pd.DataFrame],
    *,
    label_col: str = "label",
    pred_col: str = "pred",
    model_col: str = "model",
    file_col: str = "file",
    # If present, these columns will be appended to the method name to define a unique system.
    system_id_cols: Optional[List[str]] = None,
    # How to collapse duplicates per file within a system:
    #   "vote" -> majority vote on y_pred;
    #   "first" -> take first row as-is.
    duplicate_file_strategy: str = "vote",
) -> Dict[Tuple[str, str], pd.DataFrame]:
    """
    Returns {(method_plus_ids, model) -> DataFrame[file, y_true, y_pred]}.
    method_plus_ids looks like "rand_df|k=1|seed=0" if k/seed columns exist; otherwise it's just the dict key.
    This guarantees ONE ROW PER FILE per system.
    """
    systems = {}
    system_id_cols = system_id_cols or []

    for method, df in dfs.items():
        # required columns
        for c in (label_col, pred_col, model_col, file_col):
            if c not in df.columns:
                raise KeyError(f"Column '{c}' missing in dataframe '{method}'")

        # auto-detect common id cols if present
        auto_cols = [c for c in ["k", "seed"] if c in df.columns]
        if system_id_cols is None:
            id_cols = auto_cols         # auto-detect only when None
        else:
            id_cols = [c for c in system_id_cols if c in df.columns]  # empty list stays empty

        # build groups by (model, optional ids)
        group_cols = [model_col] + id_cols
        for keys, g in df.groupby(group_cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            model = keys[0]
            suffix = "".join([f"|{col}={val}" for col, val in zip(id_cols, keys[1:])])
            method_name = f"{method}{suffix}" if suffix else method

            tbl = g[[file_col, label_col, pred_col]].rename(
                columns={file_col: "file", label_col: "y_true", pred_col: "y_pred"}
            ).reset_index(drop=True)

            # collapse duplicates per file within this system
            tbl = _dedupe_by_file(tbl, agg_mode=duplicate_file_strategy)
            systems[(method_name, model)] = tbl

    return systems

# ----------------------------
# Performance table (uses unique systems)
# ----------------------------
def build_performance_table(
    dfs: Dict[str, pd.DataFrame],
    *,
    metric: str | Callable = "f1",
    average: str = "macro",
    labels: Optional[Iterable[str]] = None,
    label_col: str = "label",
    pred_col: str = "pred",
    model_col: str = "model",
    file_col: str = "file",
    system_id_cols: Optional[List[str]] = None,
    duplicate_file_strategy: str = "vote",
) -> pd.DataFrame:
    systems = _extract_system_tables(
        dfs, label_col=label_col, pred_col=pred_col, model_col=model_col, file_col=file_col,
        system_id_cols=system_id_cols, duplicate_file_strategy=duplicate_file_strategy
    )
    metric_fn = get_metric_fn(metric, average=average, labels=labels) if isinstance(metric, str) else metric

    rows = []
    for (method, model), tbl in systems.items():
        score = metric_fn(tbl["y_true"], tbl["y_pred"])
        rows.append({"model": model, "method": method, "score": float(score)})
    perf = pd.DataFrame(rows).pivot(index="model", columns="method", values="score").sort_index()
    return perf

# ----------------------------
# Paired bootstrap with tqdm (UPDATED)
# ----------------------------
def paired_bootstrap_pvalues(
    dfs: Dict[str, pd.DataFrame],
    *,
    B: int = 5000,
    stratified: bool = False,
    random_state: int = 42,
    metric: str | Callable = "f1",
    average: str = "macro",
    labels: Optional[Iterable[str]] = None,
    label_col: str = "label",
    pred_col: str = "pred",
    model_col: str = "model",
    file_col: str = "file",
    system_id_cols: Optional[List[str]] = None,
    duplicate_file_strategy: str = "vote",
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    systems = _extract_system_tables(
        dfs, label_col=label_col, pred_col=pred_col, model_col=model_col, file_col=file_col,
        system_id_cols=system_id_cols, duplicate_file_strategy=duplicate_file_strategy
    )
    sys_keys = list(systems.keys())
    metric_fn = get_metric_fn(metric, average=average, labels=labels) if isinstance(metric, str) else metric
    idx = pd.MultiIndex.from_tuples(sys_keys, names=["method", "model"])
    pvals = pd.DataFrame(np.nan, index=idx, columns=idx, dtype=float)
    total_pairs = len(sys_keys) * (len(sys_keys) - 1) // 2
    for (k1, k2) in tqdm(combinations(sys_keys, 2), total=total_pairs, desc="Bootstrap pairs"):
        a = systems[k1]
        b = systems[k2]
        # align on intersection of unique files
        common = pd.Index(a["file"]).intersection(pd.Index(b["file"]))
        if len(common) == 0:
            continue
        a = a.set_index("file").loc[common]
        b = b.set_index("file").loc[common]
        # now order is identical by index
        a = a.reset_index()
        b = b.reset_index()
        N = len(a)
        # Convert to NumPy arrays for faster slicing
        y_true_arr = a["y_true"].to_numpy()
        y_pred_a_arr = a["y_pred"].to_numpy()
        y_pred_b_arr = b["y_pred"].to_numpy()
        # observed
        obs = metric_fn(y_true_arr, y_pred_a_arr) - metric_fn(y_true_arr, y_pred_b_arr)
        # pre-generate all bootstrap indices vectorized
        if stratified:
            groups = {}
            for i, cls in enumerate(y_true_arr):
                groups.setdefault(cls, []).append(i)
            g_sizes = {c: len(ix) for c, ix in groups.items()}
            all_chosen = np.empty((B, N), dtype=int)
            col_start = 0
            for c, ix in groups.items():
                g = g_sizes[c]
                ix_arr = np.array(ix)
                # vectorized choice for all B
                choices = rng.choice(ix_arr, size=(B, g), replace=True)
                all_chosen[:, col_start : col_start + g] = choices
                col_start += g
            # shuffle each bootstrap sample
            for bi in range(B):
                rng.shuffle(all_chosen[bi])
        else:
            # vectorized integers for all B
            all_chosen = rng.integers(0, N, size=(B, N))
        diffs = np.empty(B, dtype=float)
        for bi in range(B):
            idxs = all_chosen[bi]
            # slice arrays instead of iloc on DF
            diffs[bi] = metric_fn(y_true_arr[idxs], y_pred_a_arr[idxs]) - metric_fn(y_true_arr[idxs], y_pred_b_arr[idxs])
        p = (min((diffs >= 0).mean(), (diffs <= 0).mean())) * 2.0
        p = float(min(1.0, max(0.0, p)))
        pvals.loc[k1, k2] = p
        pvals.loc[k2, k1] = p
    for k in sys_keys:
        pvals.loc[k, k] = 0.0
    return pvals

# ----------------------------
# Wrapper (UPDATED to pass new args)
# ----------------------------
def evaluate_with_significance(
    dfs: Dict[str, pd.DataFrame],
    *,
    metric: str | Callable = "f1",
    average: str = "macro",
    labels: Optional[Iterable[str]] = None,
    B: int = 5000,
    stratified: bool = False,
    random_state: int = 42,
    label_col: str = "label",
    pred_col: str = "pred",
    model_col: str = "model",
    file_col: str = "file",
    system_id_cols: Optional[List[str]] = None,
    duplicate_file_strategy: str = "vote",
):
    perf = build_performance_table(
        dfs, metric=metric, average=average, labels=labels,
        label_col=label_col, pred_col=pred_col, model_col=model_col, file_col=file_col,
        system_id_cols=system_id_cols, duplicate_file_strategy=duplicate_file_strategy
    )
    pvals = paired_bootstrap_pvalues(
        dfs, B=B, stratified=stratified, random_state=random_state, metric=metric, average=average, labels=labels,
        label_col=label_col, pred_col=pred_col, model_col=model_col, file_col=file_col,
        system_id_cols=system_id_cols, duplicate_file_strategy=duplicate_file_strategy
    )
    return perf, pvals
