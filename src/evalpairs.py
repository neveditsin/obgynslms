import numpy as np
import pandas as pd
from typing import Callable, Dict, Iterable, List, Optional, Tuple
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

try:
    from tqdm import tqdm
except ImportError:
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
# Helpers to make systems unique per file (unchanged)
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
# Core extraction (unchanged)
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
    # "vote" -> majority vote on y_pred;
    # "first" -> take first row as-is.
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
            id_cols = auto_cols  # auto-detect only when None
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
# Performance table (unchanged)
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
# New function for medical vs general pairs
# ----------------------------
def evaluate_medical_vs_general(
    dfs: Dict[str, pd.DataFrame],
    pairs: Dict[str, str],  # medical_model_name -> general_model_name
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
    """
    Evaluates performance and significance for specified medical vs general model pairs.
    - Computes full performance table for all models/methods.
    - Computes bootstrap p-values only for matching (method, medical) vs (method, general) pairs.
    - Returns perf (as before), and pair_pvals DataFrame with columns: medical_score, general_score, delta, p_value
      indexed by (pair, method).
    - Assumes model names match exactly as in the data (e.g., 'BioMistral-7B' vs 'Mistral-7B').
    - If system_id_cols like ['seed'] are used, methods will include suffixes like '|seed=0', and comparisons
      are done within the same extended method (e.g., same seed).
    """
    perf = build_performance_table(
        dfs, metric=metric, average=average, labels=labels,
        label_col=label_col, pred_col=pred_col, model_col=model_col, file_col=file_col,
        system_id_cols=system_id_cols, duplicate_file_strategy=duplicate_file_strategy
    )
    rng = np.random.default_rng(random_state)
    systems = _extract_system_tables(
        dfs, label_col=label_col, pred_col=pred_col, model_col=model_col, file_col=file_col,
        system_id_cols=system_id_cols, duplicate_file_strategy=duplicate_file_strategy
    )
    metric_fn = get_metric_fn(metric, average=average, labels=labels) if isinstance(metric, str) else metric
    # Collect unique methods (including any id suffixes)
    methods = set(method for method, model in systems)
    rows = []
    total_pairs = len(pairs) * len(methods)
    progress = tqdm(total=total_pairs, desc="Bootstrap medical-general pairs")
    for medical, general in pairs.items():
        for method in methods:
            k_med = (method, medical)
            k_gen = (method, general)
            if k_med not in systems or k_gen not in systems:
                progress.update(1)
                continue
            a = systems[k_med]  # medical
            b = systems[k_gen]  # general
            # align on intersection of unique files
            common = pd.Index(a["file"]).intersection(pd.Index(b["file"]))
            if len(common) == 0:
                progress.update(1)
                continue
            a = a.set_index("file").loc[common].reset_index()
            b = b.set_index("file").loc[common].reset_index()
            N = len(a)
            # Convert to NumPy arrays for faster slicing
            y_true_arr = a["y_true"].to_numpy()
            y_pred_a_arr = a["y_pred"].to_numpy()
            y_pred_b_arr = b["y_pred"].to_numpy()
            # observed delta: medical - general
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
            # two-sided p-value
            p = (min((diffs >= 0).mean(), (diffs <= 0).mean())) * 2.0
            p = float(min(1.0, max(0.0, p)))
            med_score = float(metric_fn(y_true_arr, y_pred_a_arr))
            gen_score = float(metric_fn(y_true_arr, y_pred_b_arr))
            rows.append({
                'pair': f'{medical} vs {general}',
                'method': method,
                'medical_score': med_score,
                'general_score': gen_score,
                'delta': obs,
                'p_value': p
            })
            progress.update(1)
    progress.close()
    pair_pvals = pd.DataFrame(rows)
    if not pair_pvals.empty:
        pair_pvals = pair_pvals.set_index(['pair', 'method']).sort_index()
    return perf, pair_pvals