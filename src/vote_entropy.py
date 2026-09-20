# vote_entropy.py
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import math
import pandas as pd



def _infer_column(df: pd.DataFrame, choices: Sequence[str]) -> Optional[str]:
    for c in choices:
        if c in df.columns:
            return c
    return None


def _ensure_dataframe(data: Union[pd.DataFrame, str]) -> pd.DataFrame:
    """
    Accept either a pandas DataFrame or a file path (csv/parquet/json).
    """
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, str):
        path = data
        lower = path.lower()
        if lower.endswith(".parquet") or lower.endswith(".pq"):
            return pd.read_parquet(path)
        if lower.endswith(".csv"):
            return pd.read_csv(path)
        if lower.endswith(".json"):
            return pd.read_json(path, lines=lower.endswith(".jsonl"))
        raise ValueError(
            f"Unsupported file extension for '{path}'. "
            "Use a DataFrame or a .parquet/.csv/.json(.jsonl) file."
        )
    raise TypeError("`data` must be a pandas DataFrame or a file path string.")


def _entropy_from_counts(counts: Iterable[int], normalize: bool) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    # Shannon entropy in bits
    h = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = c / total
        h -= p * math.log2(p)
    if not normalize:
        return h
    # normalize by max entropy (log2 K)
    k = sum(1 for c in counts if c > 0)
    return 0.0 if k <= 1 else (h / math.log2(k))


def compute_vote_entropy(
    data: Union[pd.DataFrame, str],
    *,
    include_models: Optional[Sequence[str]] = None,
    normalize_entropy: bool = True,
    # You can override these if your column names differ
    id_col: Optional[str] = None,
    model_col: Optional[str] = None,
    label_col: Optional[str] = None,
    min_votes: int = 1,
) -> pd.DataFrame:
    """
    Compute a vote-entropy table over models' predictions for each item.

    Parameters
    ----------
    data
        Either a pandas DataFrame or a path to a .parquet/.csv/.json(.jsonl) file.
    include_models
        Optional list of model names to include; if None, all models are used.
    normalize_entropy
        If True, return entropy normalized to [0, 1] by dividing by log2(K),
        where K is the number of distinct labels voted for on that item.
    id_col, model_col, label_col
        Column names. If omitted, the function will try to infer them using
        common names used in “semantic normalization” notebooks:
        - id: ['id','question_id','qid','item_id','example_id','uid','prompt_id']
        - model: ['model','model_name','provider','engine']
        - label: ['normalized','normalized_p','label','answer','answer_norm','target']
    min_votes
        Drop items with fewer than `min_votes` votes (after filtering to `include_models`).

    Returns
    -------
    pd.DataFrame
        vote_entropy_table with one row per item, containing:
        - <id_col>: item identifier
        - n_models: number of model votes used
        - vote_entropy: Shannon vote entropy (normalized if `normalize_entropy=True`)
        - top_label: most-voted label for the item
        - top_count: number of votes for the top label
        - margin: (top_count / n_models) - (second_count / n_models)  (0 if only one label)
        - counts: JSON-like dict of {label: count} (object dtype)
    """
    df = _ensure_dataframe(data)

    # Infer columns if not provided
    id_col = id_col or _infer_column(
        df, ["id", "question_id", "qid", "item_id", "example_id", "uid", "prompt_id"]
    )
    model_col = model_col or _infer_column(df, ["model", "model_name", "provider", "engine"])
    label_col = label_col or _infer_column(
        df, ["normalized", "normalized_p", "label", "answer", "answer_norm", "target"]
    )

    missing = [name for name, val in [("id_col", id_col), ("model_col", model_col), ("label_col", label_col)] if val is None]
    if missing:
        raise ValueError(
            "Could not infer required column(s): "
            + ", ".join(missing)
            + ". Please pass them explicitly (e.g., id_col='qid', model_col='model', label_col='normalized')."
        )

    work = df[[id_col, model_col, label_col]].dropna(subset=[id_col, model_col, label_col])

    if include_models is not None:
        include_set = set(include_models)
        work = work[work[model_col].isin(include_set)]

    # Aggregate votes per item
    groups = work.groupby(id_col, sort=False)

    rows: List[dict] = []
    for item_id, g in groups:
        labels = g[label_col].astype(str).tolist()
        if len(labels) < min_votes:
            continue

        # Count votes per label
        c = Counter(labels)
        counts_dict = dict(c)

        # Compute entropy
        h = _entropy_from_counts(c.values(), normalize=normalize_entropy)

        # Top label, margin
        rank: List[Tuple[str, int]] = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))
        top_label, top_count = rank[0]
        second_count = rank[1][1] if len(rank) > 1 else 0
        n = len(labels)
        margin = (top_count / n) - (second_count / n)

        rows.append(
            {
                id_col: item_id,
                "n_models": n,
                "vote_entropy": h,
                "top_label": top_label,
                "top_count": top_count,
                "margin": margin,
                "counts": counts_dict,
            }
        )

    out = pd.DataFrame(rows).sort_values([id_col]).reset_index(drop=True)
    return out




# --- add to vote_entropy.py ---

# --- replace the earlier helpers with these versions ---

from pathlib import Path
from typing import Callable, Dict, List, Tuple, Optional
import pandas as pd
from typing import Any, Callable, Dict, List, Optional, Tuple
import pandas as pd
from pathlib import Path

def topk_by_top_label(
    vote_entropy_table: pd.DataFrame,
    *,
    k: int = 3,
    id_col: str = "file",
    top_label_col: str = "top_label",
    entropy_col: str = "vote_entropy",
    text_loader: Optional[Callable[[str], str]] = None,
    directory: Optional[str] = None,
    # gold matching
    gold_df: Optional[pd.DataFrame] = None,
    gold_on: str = "file",
    gold_label_col: str = "label",
) -> Dict[str, List[Tuple[str, str, float, Optional[str]]]]:
    """
    For each distinct `top_label`, return up to `k` docs with highest vote entropy.
    Returns: label -> [(file_name, text, vote_entropy, gold_label_or_None), ...]

    Matching rule for gold labels is identical to `evaluate_by_model`:
    - For each input file name X, find the first g in gold_df[gold_on] such that (g in X).
    - If found, gold label is gold_df.loc[gold_on == g, gold_label_col].iloc[0]; else None.
    """

    # --- Basic validation ---
    required_cols = [id_col, top_label_col, entropy_col]
    missing = [c for c in required_cols if c not in vote_entropy_table.columns]
    if missing:
        raise ValueError(f"Missing required columns in vote_entropy_table: {missing}")

    # --- Text loader (default reads from directory) ---
    def _default_loader(fname: str) -> str:
        if directory is None:
            raise ValueError(
                "No text_loader provided and directory=None. "
                "Pass a `text_loader` or a `directory` to read from."
            )
        p = Path(directory) / str(fname)
        if not p.exists():
            return f"[{fname} not found in {directory}]"
        try:
            return p.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            return f"[ERROR reading {fname}: {e}]"

    loader = text_loader or _default_loader

    # --- Prepare gold lookup with substring logic (g in file_name) ---
    if gold_df is not None:
        if gold_on not in gold_df.columns or gold_label_col not in gold_df.columns:
            raise ValueError(f"gold_df must contain columns '{gold_on}' and '{gold_label_col}'")
        # Keep original order and cast to str for robust substring matching.
        _gold_names: List[str] = [str(x) for x in gold_df[gold_on].tolist()]
        # Map gold name -> label for quick retrieval
        _gold_label_map = dict(
            zip([str(x) for x in gold_df[gold_on].tolist()],
                [str(x) for x in gold_df[gold_label_col].tolist()])
        )

        def best_match_label(file_name: str) -> Optional[str]:
            s = str(file_name)
            match = next((g for g in _gold_names if g in s), None)
            return _gold_label_map.get(match) if match is not None else None
    else:
        def best_match_label(_: str) -> Optional[str]:
            return None

    # --- Sort: group by top_label, pick top-k by entropy desc ---
    if len(vote_entropy_table) == 0:
        return {}

    sorted_df = vote_entropy_table.sort_values(
        [top_label_col, entropy_col],
        ascending=[True, False],
        kind="stable",
    )

    out: Dict[str, List[Tuple[str, str, float, Optional[str]]]] = {}
    for label, group in sorted_df.groupby(top_label_col, sort=False):
        picks = group.head(k)
        tuples: List[Tuple[str, str, float, Optional[str]]] = []
        for _, row in picks.iterrows():
            fname = str(row[id_col])
            ent = float(row[entropy_col])
            # Load text safely
            try:
                text = loader(fname)
            except Exception as e:
                text = f"[ERROR loading {fname}: {e}]"
            gold_label = best_match_label(fname)
            tuples.append((fname, text, ent, gold_label))
        out[str(label)] = tuples

    return out


def print_topk_by_top_label(
    results: Dict[str, List[Tuple[str, str, float, Optional[str]]]],
    *,
    max_chars: int = 800
) -> None:
    """
    Plain-text printing of `topk_by_top_label` results.
    """
    for label, items in results.items():
        print(f"Top-label group: {label}")
        if not items:
            print("(no items)\n")
            continue
        for i, (fname, text, ent, gold_label) in enumerate(items, start=1):
            snippet = (text[:max_chars] + "…") if len(text) > max_chars else text
            print(f"  {i}. File: {fname}")
            print(f"     Entropy: {ent:.3f}")
            print(f"     Label: {gold_label if gold_label is not None else 'N/A'}")  # gold
            print("     Text:")
            # indent snippet lines without using f-string backslash tricks
            for line in snippet.splitlines():
                print(f"       {line}")
            print()  # blank line between items
        print()      # blank line between groups

def make_icl_prompt_from_results(
    results: Dict[str, List[Tuple[str, str, float, Optional[str]]]],
    *,
    instruction: str = (
        "You are a medical assistant specialized in obstetric ultrasound.\n"
        "Classify the document-level active pregnancy status as one of:\n\n"
        "early active pregnancy\n"
        "late active pregnancy\n"
        "no active pregnancy\n\n"
        "Only output the label, nothing else."
    ),
    input_tag: str = "Report",
    output_tag: str = "Label",
    truncate_chars: int = 1200,
    include_group_headers: bool = False,
    use_gold_if_available: bool = False,
    add_query_placeholder: bool = True,
    include_rationales: bool = False,
    rationale_tag: str = "Rationale",
    rationale_lookup: Optional[Dict[str, str]] = None,
) -> str:
    """
    Build a few-shot prompt using *all* examples contained in `results`,
    which should be the output of `topk_by_top_label`.

    Parameters
    ----------
    results
        Mapping: top_label -> [(file, text, entropy, gold_label_or_None), ...]
    instruction
        Instruction line placed at the top of the prompt.
    input_tag, output_tag
        Field labels for examples, e.g., "Report", "Label".
    truncate_chars
        Max characters of each example's text.
    include_group_headers
        If True, insert a '### Top-label group: <label>' header for each group.
    use_gold_if_available
        If True, use the gold label when present in results tuples; else use the group label.
    add_query_placeholder
        If True, append a final classification slot for a new input ({{report_text}}).
    include_rationales
        If True, append a rationale line for each exemplar using ``rationale_lookup``.

    Returns
    -------
    str
        Ready-to-use few-shot prompt.
    """
    lines: List[str] = []
    instruction_block = instruction.strip()
    if instruction_block:
        lines.append(instruction_block)
        lines.append("")

    # Deterministic group ordering
    for group_label in sorted(results.keys()):
        items = results[group_label] or []
        if include_group_headers:
            lines.append(f"### Top-label group: {group_label}")
            lines.append("")

        label_map = {
            "early": "early active pregnancy",
            "early pregnancy": "early active pregnancy",
            "early active pregnancy": "early active pregnancy",
            "late": "late active pregnancy",
            "late pregnancy": "late active pregnancy",
            "late active pregnancy": "late active pregnancy",
            "unrelated": "no active pregnancy",
            "unrelated or no current pregnancy": "no active pregnancy",
            "no current pregnancy": "no active pregnancy",
            "no active pregnancy": "no active pregnancy",
            "no": "no active pregnancy",
        }

        for idx, (fname, text, _ent, gold_label) in enumerate(items, start=1):
            base_label = gold_label if (use_gold_if_available and gold_label is not None) else group_label
            # canonicalize the label we will render
            base_key = str(base_label).strip().lower() if base_label is not None else ""
            label_out = label_map.get(base_key, str(base_label))
            snippet = text[:truncate_chars] if isinstance(text, str) else ""
            snippet_clean = snippet.strip()
            if snippet_clean:
                lines.append(f"{input_tag}: {snippet_clean}")
            else:
                lines.append(f"{input_tag}:")
            # Use resolved label_out to avoid emitting 'None'
            lines.append(f"{output_tag}: {label_out}")
            if include_rationales:
                rationale = None
                if rationale_lookup is not None:
                    rationale = rationale_lookup.get(str(fname))
                if rationale is not None and str(rationale).strip():
                    lines.append(f"{rationale_tag}: {str(rationale).strip()}")
            lines.append("")  # blank line between examples

    if add_query_placeholder:
        lines.append("NOW YOUR TURN")
        lines.append(f"{input_tag}: {{report_text}}")
        lines.append(f"{output_tag}:")

    return "\n".join(lines).strip()
















from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Optional dependency: sentence-transformers
# pip install sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


def _minmax_normalize(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    amin, amax = float(np.min(arr)), float(np.max(arr))
    if amax == amin:
        # all equal → return ones to avoid killing the signal
        return np.ones_like(arr, dtype=float)
    return (arr - amin) / (amax - amin)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    # expects 1D vectors
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def build_e5_embedder(model_name: str = "intfloat/multilingual-e5-large",
                      device: Optional[str] = None,
                      prefix: str = "passage: ") -> Callable[[List[str]], np.ndarray]:
    """
    Returns a callable: embedder(texts: List[str]) -> np.ndarray [len(texts), dim]
    Uses multilingual-e5; prefixes each text with 'passage: ' by default.
    """
    if SentenceTransformer is None:
        raise ImportError("sentence-transformers not available. Install it to use E5 embeddings.")
    model = SentenceTransformer(model_name, device=device)

    def _embedder(texts: List[str]) -> np.ndarray:
        if prefix:
            texts = [f"{prefix}{t}" for t in texts]
        embs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(embs, dtype=float)

    return _embedder


def cbed_select_by_label(
    vote_entropy_table: pd.DataFrame,
    *,
    k_per_label: int = 3,
    id_col: str = "file",
    top_label_col: str = "top_label",
    entropy_col: str = "vote_entropy",
    text_loader: Optional[Callable[[str], str]] = None,
    directory: Optional[str] = None,
    # embeddings
    embedder: Optional[Callable[[List[str]], np.ndarray]] = None,
    e5_model_name: str = "intfloat/multilingual-e5-large",
    e5_device: Optional[str] = None,
    e5_prefix: str = "passage: ",
    embedding_cache: Optional[Dict[str, np.ndarray]] = None,
    # gold matching (substring logic)
    gold_df: Optional[pd.DataFrame] = None,
    gold_on: str = "file",
    gold_label_col: str = "label",
) -> Dict[str, List[Tuple[str, str, float, Optional[str]]]]:
    """
    Class-Balanced Entropy–Diversity (CBED) selection using multilingual-e5 embeddings.

    For each distinct `top_label`, select up to `k_per_label` exemplars by:
      1) Seed: highest normalized vote entropy in that label.
      2) Greedy: at each step, compute for each remaining candidate i:
           H' = min–max normalized entropy (within-label)
           D  = 1 - max_{j in S} cos(e[i], e[j])   (max dissimilarity to selected set)
         Score(i) = harmonic_mean(H', D) = 2*H'*D / (H'+D)
         pick argmax Score.

    Returns:
      label -> [(file_name, text, vote_entropy, gold_label_or_None), ...] (len ≤ k_per_label)
    """
    # --- Basic validation ---
    required = [id_col, top_label_col, entropy_col]
    missing = [c for c in required if c not in vote_entropy_table.columns]
    if missing:
        raise ValueError(f"Missing required columns in vote_entropy_table: {missing}")
    if vote_entropy_table.empty:
        return {}

    # --- Text loader (default reads from directory) ---
    def _default_loader(fname: str) -> str:
        if directory is None:
            raise ValueError(
                "No text_loader provided and directory=None. "
                "Pass a `text_loader` or a `directory` to read from."
            )
        p = Path(directory) / str(fname)
        if not p.exists():
            return f"[{fname} not found in {directory}]"
        try:
            return p.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            return f"[ERROR reading {fname}: {e}]"

    loader = text_loader or _default_loader

    # --- Gold lookup (substring match) ---
    if gold_df is not None:
        if gold_on not in gold_df.columns or gold_label_col not in gold_df.columns:
            raise ValueError(f"gold_df must contain '{gold_on}' and '{gold_label_col}'")
        _gold_names: List[str] = [str(x) for x in gold_df[gold_on].tolist()]
        _gold_label_map = dict(
            zip([str(x) for x in gold_df[gold_on].tolist()],
                [str(x) for x in gold_df[gold_label_col].tolist()])
        )

        def best_match_label(file_name: str) -> Optional[str]:
            s = str(file_name)
            match = next((g for g in _gold_names if g in s), None)
            return _gold_label_map.get(match) if match is not None else None
    else:
        def best_match_label(_: str) -> Optional[str]:
            return None

    # --- Prepare embedder if not provided ---
    _embedder = embedder
    if embedding_cache is None:
        if _embedder is None:
            _embedder = build_e5_embedder(e5_model_name, device=e5_device, prefix=e5_prefix)
    
    # --- Work per label ---
    out: Dict[str, List[Tuple[str, str, float, Optional[str]]]] = {}
    for label, group in vote_entropy_table.groupby(top_label_col, sort=False):
        # Sort by entropy desc for deterministic seeding, but we’ll compute normalized within this group
        grp = group[[id_col, entropy_col]].copy().reset_index(drop=True)
        ids = grp[id_col].astype(str).tolist()
        entropies = grp[entropy_col].astype(float).to_numpy()

        # Normalize entropy within label
        Hn = _minmax_normalize(entropies)  # shape [n_c]

        # Load texts once
        texts = []
        for fname in ids:
            try:
                texts.append(loader(fname))
            except Exception as e:
                texts.append(f"[ERROR loading {fname}: {e}]")

        # Pre-compute embeddings for all candidates in this label
        if embedding_cache is not None:
            try:
                embs = np.stack([embedding_cache[fname] for fname in ids])
            except KeyError as exc:
                missing = exc.args[0]
                raise KeyError(f"Embedding cache missing vector for '{missing}'") from exc
        else:
            embs = _embedder(texts)  # [n_c, dim]; normalized if embedder sets normalize_embeddings=True
            if not isinstance(embs, np.ndarray):
                embs = np.asarray(embs)

        n_c = len(ids)
        if n_c == 0:
            out[str(label)] = []
            continue

        selected_idx: List[int] = []

        # 1) Seed: highest normalized entropy
        seed = int(np.argmax(Hn))
        selected_idx.append(seed)

        # 2) Greedy until k_per_label
        while len(selected_idx) < min(k_per_label, n_c):
            best_i, best_score = None, -1.0
            # Precompute selected embeddings
            sel_embs = embs[selected_idx, :]

            for i in range(n_c):
                if i in selected_idx:
                    continue

                # Dissimilarity wrt selected set: 1 - max cosine sim
                if len(selected_idx) == 0:
                    D = 1.0
                else:
                    sims = np.dot(sel_embs, embs[i])  # since normalized
                    max_sim = float(np.max(sims)) if sims.size > 0 else 0.0
                    # guard numeric wobble
                    max_sim = max(min(max_sim, 1.0), -1.0)
                    D = 1.0 - max_sim  # in [0,1] when embs are normalized

                H = float(Hn[i])

                # Harmonic mean (parameter-free); avoid division by zero
                denom = (H + D)
                score = 0.0 if denom == 0.0 else (2.0 * H * D) / denom

                # Tie-breaks: prefer higher H, then higher D, then lower index
                if (score > best_score or
                   (score == best_score and (H > float(Hn[best_i])) if best_i is not None else False) or
                   (score == best_score and H == float(Hn[best_i]) if best_i is not None else False and D > 0)):
                    best_i, best_score = i, score

            if best_i is None:
                break
            selected_idx.append(int(best_i))

        # Build output tuples for this label
        tuples: List[Tuple[str, str, float, Optional[str]]] = []
        for idx in selected_idx:
            fname = ids[idx]
            text = texts[idx]
            ent = float(entropies[idx])
            gold_label = best_match_label(fname)
            tuples.append((fname, text, ent, gold_label))

        out[str(label)] = tuples
    # --- Cleanup: remove model and free cache ---
    try:
        if hasattr(_embedder, "__self__") and hasattr(_embedder.__self__, "model"):
            del _embedder.__self__.model
        elif hasattr(_embedder, "model"):
            del _embedder.model
    except Exception:
        pass

    try:
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
        
    return out









from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

def randomk_by_gold_label(
    vote_entropy_table: pd.DataFrame,
    *,
    k: int = 3,
    id_col: str = "file",
    # we ignore top_label_col and entropy for selection; entropy is only returned if present
    entropy_col: str = "vote_entropy",
    text_loader: Optional[Callable[[str], str]] = None,
    directory: Optional[str] = None,
    # gold matching (REQUIRED here)
    gold_df: Optional[pd.DataFrame] = None,
    gold_on: str = "file",
    gold_label_col: str = "label",
    # reproducibility
    seed: Optional[int] = None,
    simulate_reveal: bool = True,
) -> Dict[str, List[Tuple[str, str, float, Optional[str]]]]:
    """
    Random selection by gold label for ICL exemplars.

    Default behavior (`simulate_reveal=True`) mirrors reveal-based random drawing:
      - draw uniformly at random without replacement from the unlabeled pool,
      - reveal gold label after each draw,
      - accept the draw only if that gold class has not yet reached quota `k`,
      - stop when all classes reach quota or the pool is exhausted.

    Legacy behavior (`simulate_reveal=False`) samples up to `k` items independently
    inside each gold-label bucket.

    Returns
    -------
    Dict[str, List[Tuple[str, str, float, Optional[str]]]]
        gold_label -> [(file_name, text, vote_entropy_or_nan, gold_label), ...]

    Notes:
      - Selection ignores vote entropy; it is returned only if present in the input table.
      - Gold label is assigned via substring logic: for a file X, find the first g in gold_df[gold_on] such that (g in X).
    """

    # --- Basic validation ---
    if id_col not in vote_entropy_table.columns:
        raise ValueError(f"Missing required column '{id_col}' in vote_entropy_table")
    if gold_df is None:
        raise ValueError("gold_df is required for random selection by gold label.")
    if gold_on not in gold_df.columns or gold_label_col not in gold_df.columns:
        raise ValueError(f"gold_df must contain columns '{gold_on}' and '{gold_label_col}'")

    rng = np.random.default_rng(seed)

    # --- Text loader (default reads from directory) ---
    def _default_loader(fname: str) -> str:
        if directory is None:
            raise ValueError(
                "No text_loader provided and directory=None. "
                "Pass a `text_loader` or a `directory` to read from."
            )
        p = Path(directory) / str(fname)
        if not p.exists():
            return f"[{fname} not found in {directory}]"
        try:
            return p.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            return f"[ERROR reading {fname}: {e}]"

    loader = text_loader or _default_loader

    # --- Prepare gold lookup with substring logic (g in file_name) ---
    _gold_names: List[str] = [str(x) for x in gold_df[gold_on].tolist()]
    _gold_label_map = dict(
        zip([str(x) for x in gold_df[gold_on].tolist()],
            [str(x) for x in gold_df[gold_label_col].tolist()])
    )

    def best_match_label(file_name: str) -> Optional[str]:
        s = str(file_name)
        match = next((g for g in _gold_names if g in s), None)
        return _gold_label_map.get(match) if match is not None else None

    if vote_entropy_table.empty:
        return {}

    # --- Build per-gold-label index lists ---
    df = vote_entropy_table[[id_col] + ([entropy_col] if entropy_col in vote_entropy_table.columns else [])].copy()
    df["_gold_label"] = df[id_col].astype(str).apply(best_match_label)

    # Keep only rows that matched some gold label
    df = df[~df["_gold_label"].isna()].reset_index(drop=True)

    if df.empty:
        return {}

    # Preserve a stable label iteration order in outputs/prompts.
    label_order = (
        df["_gold_label"]
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    out: Dict[str, List[Tuple[str, str, float, Optional[str]]]] = {g: [] for g in label_order}

    if not simulate_reveal:
        # Legacy per-label independent sampling.
        label_to_indices: Dict[str, List[int]] = {}
        for idx, row in df.iterrows():
            glabel = str(row["_gold_label"])
            label_to_indices.setdefault(glabel, []).append(idx)

        for glabel, idxs in label_to_indices.items():
            n = len(idxs)
            if n <= 0:
                continue
            sample_size = min(int(k), n)
            chosen = list(rng.choice(idxs, size=sample_size, replace=False))
            for i in chosen:
                fname = str(df.loc[i, id_col])
                ent = float(df.loc[i, entropy_col]) if entropy_col in df.columns else float("nan")
                try:
                    text = loader(fname)
                except Exception as e:
                    text = f"[ERROR loading {fname}: {e}]"
                out[glabel].append((fname, text, ent, glabel))
        return out

    # Reveal-based random draw without replacement from the unlabeled pool.
    quota = max(0, int(k))
    accepted_counts: Dict[str, int] = {g: 0 for g in label_order}

    if quota == 0:
        return out

    draw_order = rng.permutation(len(df))
    for i in draw_order:
        idx = int(i)
        glabel = str(df.loc[idx, "_gold_label"])
        if accepted_counts[glabel] >= quota:
            continue

        fname = str(df.loc[idx, id_col])
        ent = float(df.loc[idx, entropy_col]) if entropy_col in df.columns else float("nan")
        try:
            text = loader(fname)
        except Exception as e:
            text = f"[ERROR loading {fname}: {e}]"

        out[glabel].append((fname, text, ent, glabel))
        accepted_counts[glabel] += 1

        if all(count >= quota for count in accepted_counts.values()):
            break

    return out







from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

def cbed_select_by_label_with_gold_random(
    vote_entropy_table: pd.DataFrame,
    *,
    k_per_label: int = 3,
    id_col: str = "file",
    top_label_col: str = "top_label",
    entropy_col: str = "vote_entropy",
    text_loader: Optional[Callable[[str], str]] = None,
    directory: Optional[str] = None,
    # embeddings
    embedder: Optional[Callable[[List[str]], np.ndarray]] = None,
    e5_model_name: str = "intfloat/multilingual-e5-large",
    e5_device: Optional[str] = None,
    e5_prefix: str = "passage: ",
    # gold matching (substring logic)
    gold_df: Optional[pd.DataFrame] = None,
    gold_on: str = "file",
    gold_label_col: str = "label",
    # reproducibility for the random gold exemplar
    seed: Optional[int] = None,
) -> Dict[str, List[Tuple[str, str, float, Optional[str]]]]:
    """
    Unified selector.

    Behavior:
      - If k_per_label == 1: identical to CBED (class-balanced entropy–diversity) per top_label.
      - If k_per_label > 1: for each label L,
            1) Pick ONE RANDOM document among rows whose gold label (via substring) == L
               (sampled from the WHOLE table, not only top_label==L).
            2) Fill the remaining (k_per_label - 1) using CBED on rows with top_label==L,
               excluding the random exemplar if it happens to be in that group.
    Returns:
      label -> [(file_name, text, vote_entropy, gold_label_or_None), ...] (len ≤ k_per_label)

    Notes:
      - Requires helper functions `_minmax_normalize` and `build_e5_embedder` available in scope.
      - Embeddings are computed only for CBED candidates (per-label group). The single random exemplar
        is not used in diversity scoring and may come from outside the CBED group.
    """

    # --- Basic validation ---
    required = [id_col, top_label_col, entropy_col]
    missing = [c for c in required if c not in vote_entropy_table.columns]
    if missing:
        raise ValueError(f"Missing required columns in vote_entropy_table: {missing}")
    if vote_entropy_table.empty:
        return {}

    rng = np.random.default_rng(seed)

    # --- Text loader (default reads from directory) ---
    def _default_loader(fname: str) -> str:
        if directory is None:
            raise ValueError(
                "No text_loader provided and directory=None. "
                "Pass a `text_loader` or a `directory` to read from."
            )
        p = Path(directory) / str(fname)
        if not p.exists():
            return f"[{fname} not found in {directory}]"
        try:
            return p.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            return f"[ERROR reading {fname}: {e}]"

    loader = text_loader or _default_loader

    # --- Gold lookup (substring match) ---
    if gold_df is not None:
        if gold_on not in gold_df.columns or gold_label_col not in gold_df.columns:
            raise ValueError(f"gold_df must contain '{gold_on}' and '{gold_label_col}'")
        _gold_names: List[str] = [str(x) for x in gold_df[gold_on].tolist()]
        _gold_label_map = dict(
            zip([str(x) for x in gold_df[gold_on].tolist()],
                [str(x) for x in gold_df[gold_label_col].tolist()])
        )

        def best_match_label(file_name: str) -> Optional[str]:
            s = str(file_name)
            match = next((g for g in _gold_names if g in s), None)
            return _gold_label_map.get(match) if match is not None else None
    else:
        def best_match_label(_: str) -> Optional[str]:
            return None

    # --- Precompute a lightweight view and gold labels per row ---
    base_cols = [id_col, top_label_col, entropy_col]
    base = vote_entropy_table[base_cols].copy()
    base[id_col] = base[id_col].astype(str)
    base["_gold_label"] = base[id_col].apply(best_match_label)

    # Fast lookup by id
    _row_by_id = {row[id_col]: row for _, row in base.iterrows()}

    # --- Prepare embedder if not provided ---
    _embedder = embedder
    if _embedder is None:
        _embedder = build_e5_embedder(e5_model_name, device=e5_device, prefix=e5_prefix)

    out: Dict[str, List[Tuple[str, str, float, Optional[str]]]] = {}

    # Helper: CBED selection for a single label group with an optional exclusion set and a target count
    def _cbed_pick_for_group(group_df: pd.DataFrame, exclude_ids: set, target_k: int) -> List[Tuple[str, str, float, Optional[str]]]:
        if target_k <= 0 or group_df.empty:
            return []

        grp = group_df[[id_col, entropy_col]].copy().reset_index(drop=True)
        # Exclude ids (e.g., the already-chosen random exemplar if present)
        if exclude_ids:
            grp = grp[~grp[id_col].isin(exclude_ids)].reset_index(drop=True)
        if grp.empty:
            return []

        ids = grp[id_col].astype(str).tolist()
        entropies = grp[entropy_col].astype(float).to_numpy()
        Hn = _minmax_normalize(entropies)

        # Load texts
        texts = []
        for fname in ids:
            try:
                texts.append(loader(fname))
            except Exception as e:
                texts.append(f"[ERROR loading {fname}: {e}]")

        # Embeddings (assumed L2-normalized by the embedder)
        embs = _embedder(texts)
        n_c = len(ids)
        if n_c == 0:
            return []

        selected_idx: List[int] = []
        # seed with highest normalized entropy
        seed_idx = int(np.argmax(Hn))
        selected_idx.append(seed_idx)

        # greedy fill
        while len(selected_idx) < min(target_k, n_c):
            best_i, best_score = None, -1.0
            sel_embs = embs[selected_idx, :]
            for i in range(n_c):
                if i in selected_idx:
                    continue
                sims = np.dot(sel_embs, embs[i]) if len(selected_idx) > 0 else np.array([])
                max_sim = float(np.max(sims)) if sims.size > 0 else 0.0
                max_sim = max(min(max_sim, 1.0), -1.0)
                D = 1.0 - max_sim
                H = float(Hn[i])
                denom = H + D
                score = 0.0 if denom == 0.0 else (2.0 * H * D) / denom
                if best_i is None or score > best_score or (score == best_score and H > float(Hn[best_i])):
                    best_i, best_score = i, score
            if best_i is None:
                break
            selected_idx.append(int(best_i))

        tuples: List[Tuple[str, str, float, Optional[str]]] = []
        for idx in selected_idx[:target_k]:
            fid = ids[idx]
            text = texts[idx]
            ent = float(entropies[idx])
            g_lbl = best_match_label(fid)
            tuples.append((fid, text, ent, g_lbl))
        return tuples

    # --- Main per-label loop ---
    for label, group in base.groupby(top_label_col, sort=False):
        label_str = str(label)

        if k_per_label == 1:
            # Pure CBED for this label
            picks = _cbed_pick_for_group(group, exclude_ids=set(), target_k=1)
            out[label_str] = picks
            continue

        # k_per_label > 1:
        # 1) random ONE from anywhere in the table with gold == current label
        random_tuple: Optional[Tuple[str, str, float, Optional[str]]] = None
        if gold_df is not None:
            pool = base[base["_gold_label"] == label_str]
            if not pool.empty:
                # sample one row index
                idx = int(rng.integers(low=0, high=len(pool)))
                row = pool.iloc[idx]
                fid = str(row[id_col])
                ent = float(row[entropy_col])
                try:
                    txt = loader(fid)
                except Exception as e:
                    txt = f"[ERROR loading {fid}: {e}]"
                random_tuple = (fid, txt, ent, label_str)

        # 2) CBED remainder within this label group (exclude random exemplar if it happens to live here)
        exclude = {random_tuple[0]} if random_tuple is not None else set()
        cbed_needed = k_per_label - (1 if random_tuple is not None else 0)
        cbed_picks = _cbed_pick_for_group(group, exclude_ids=exclude, target_k=cbed_needed)

        # Combine (random first, then CBED)
        combined: List[Tuple[str, str, float, Optional[str]]] = []
        if random_tuple is not None:
            combined.append(random_tuple)
        combined.extend(cbed_picks)

        # If no random was available (no gold match for this label), you’ll just get CBED(k_per_label)
        if random_tuple is None and len(combined) == 0:
            # nothing selected
            out[label_str] = []
        else:
            out[label_str] = combined[:k_per_label]

    # --- Cleanup: free model/cuda cache if possible ---
    if embedding_cache is None and _embedder is not None:
        try:
            if hasattr(_embedder, "__self__") and hasattr(_embedder.__self__, "model"):
                del _embedder.__self__.model
            elif hasattr(_embedder, "model"):
                del _embedder.model
        except Exception:
            pass
        try:
            import gc, torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    return out



from typing import Callable, Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
import pandas as pd


def select_icls_pingpong(
    table: pd.DataFrame,
    *,
    k: int,
    # columns
    id_col: str = "file",
    pred_col: str = "top_label",
    entropy_col: str = "vote_entropy",
    # text loading
    text_loader: Optional[Callable[[str], str]] = None,
    directory: Optional[str] = None,
    # embeddings
    embedder: Optional[Callable[[List[str]], np.ndarray]] = None,
    e5_model_name: str = "intfloat/multilingual-e5-large",
    e5_device: Optional[str] = None,
    e5_prefix: str = "passage: ",
    embedding_cache: Optional[Dict[str, np.ndarray]] = None,
    # gold matching via substring
    gold_df: Optional[pd.DataFrame] = None,
    gold_on: str = "file",
    gold_label_col: str = "label",
    # scoring and selection behavior
    score: str = "harmonic",          # "harmonic" | "sum" | "product" | "uncertainty" | "diversity"
    alpha: float = 0.5,               # used when score == "sum"
    use_random_pong: bool = True,     # if False, disable "pong" turns and always run score-based ping turns
    pong_score: str = "random",       # "random" | "harmonic" | "sum" | "product" | "uncertainty" | "diversity"
    use_pred_round_robin: bool = True,
    global_pool_round_local_diversity_after_first: bool = False,
    max_consecutive_pred: int = 2,    # cap consecutive VE picks from same predicted class
    per_label: bool = True,           # if True, treat k as per-gold-label quota
    pong_restore_on_mismatch_only: bool = False,  # if True, run random pong only after a round with ping mismatches
    pong_restore_after_each_ping: bool = False,   # if True, attempt random restore after each ping (post-warmup)
    pong_restore_warmup_pings: int = 0,           # minimum number of ping picks before per-ping restore is allowed
    reveal_one_by_one: bool = False,              # if True, use reveal-after-pick balancing policy (per-label mode only)
    reveal_one_by_one_gap_threshold: int = 2,     # early compensation trigger when max(count)-min(count) >= threshold
    reveal_no_pong_compensation: bool = False,    # if True, disable reveal compensation and discard over-quota revealed picks
    reveal_ping_pred_round_robin: bool = False,   # if True, reveal ping cycles predicted-label buckets and takes next item (no score)
    reveal_ping_pred_round_robin_prob_update: bool = False,  # if True, reveal ping samples by class-prob from counts and updates probs on mismatch
    reveal_ping_pred_deficit_h: bool = False,     # if True, reveal ping targets predicted label of largest gold deficit and picks max precomputed H
    reveal_ping_random_pool: bool = False,        # if True, reveal ping ignores scores and samples uniformly from remaining unlabeled pool
    reveal_ping_alternate_uncertainty_diversity: bool = False,  # if True, reveal ping alternates H-only and diversity-only picks
    granule_prior_round_robin: bool = False,      # if True, use granule-prior guided reveal round-robin without pong compensation
    granule_prior_cap: float = 0.95,              # cap for boosted revealed-class prior inside selected granule
    # reproducibility
    seed: Optional[int] = None,
    # optional telemetry sink (mutated in-place)
    stats_out: Optional[Dict[str, Any]] = None,
) -> List[Tuple[str, str, float, Optional[str], Optional[str]]]:
    """
    Select few-shot exemplars via score-based "ping" turns, optionally alternating with
    random-gold "pong" turns.

    Parameters
    ----------
    table : pd.DataFrame
        Vote-entropy table containing at least id/prediction/entropy columns.
    k : int
        Number of exemplars to select. When ``per_label`` is True (default), ``k`` is interpreted
        as the quota per gold class. When ``per_label`` is False, ``k`` is treated as a global
        budget.
    per_label : bool, optional
        If True, enforce up to ``k`` examples per gold label (requires ``gold_df``). When False,
        the algorithm behaves like the original global-k selector. If a class has fewer than ``k``
        eligible items, the available examples are returned.
    pong_restore_on_mismatch_only : bool, optional
        If True, treat selection as ping rounds (one per predicted-majority bucket) and invoke
        random pong restoration only when at least one ping in that round has predicted label
        different from gold label.
    pong_restore_after_each_ping : bool, optional
        If True, attempt random class-balance restoration after each ping pick
        (subject to ``pong_restore_warmup_pings``).
    pong_restore_warmup_pings : int, optional
        Minimum number of ping picks before per-ping restoration can start.
    reveal_one_by_one : bool, optional
        If True (and ``per_label`` with ``gold_df`` is enabled), simulate drawing from an
        unlabeled pool one-by-one: ping picks cannot use gold labels before selection. After each
        pick, the gold label is "revealed" and class counts are updated. Random pong turns are used
        as compensation to maintain final class balance.
    reveal_one_by_one_gap_threshold : int, optional
        Early compensation trigger for reveal-one-by-one mode. When
        ``max(counts_gold) - min(counts_gold) >= threshold``, choose random compensation from
        deficit classes (default: 2).
    reveal_no_pong_compensation : bool, optional
        Only affects reveal-one-by-one mode. When True, never run reveal compensation picks.
        Instead, keep making ping picks from the unlabeled pool until per-class quotas are met
        (or data is exhausted). Revealed picks from already-full classes are discarded.
    reveal_ping_pred_round_robin : bool, optional
        Only affects reveal-one-by-one mode. When True, ping picks target predicted-label
        buckets in strict round-robin order and choose the next available candidate in that
        bucket (no uncertainty/diversity scoring inside the bucket).
    reveal_ping_pred_round_robin_prob_update : bool, optional
        Only affects reveal-one-by-one mode. When True, ping targets gold classes in
        round-robin (skipping filled quotas), then samples from the remaining pool with
        probabilities proportional to per-document class probabilities from vote ``counts``
        (fallback: one-hot on ``pred_col``). After reveal, if sampled class intent and gold
        mismatch, probabilities of similar remaining documents are updated by moving mass from
        intended class to revealed class.
    reveal_ping_pred_deficit_h : bool, optional
        Only affects reveal-one-by-one mode. When True, ping picks first target the currently
        largest revealed gold-class deficit, map that deficit class to the candidate predicted
        label (majority vote ``pred_col``), and choose the remaining candidate with the highest
        precomputed normalized entropy ``_H`` in that predicted-label bucket. If no remaining
        candidate matches the top deficit class, the selector tries the next deficit class, then
        falls back to the standard reveal ping score rule if no deficit-label bucket is available.
    reveal_ping_random_pool : bool, optional
        Only affects reveal-one-by-one mode. When True, ping picks are sampled uniformly at
        random from the whole remaining unlabeled pool (no score), while reveal compensation
        behavior remains unchanged.
    reveal_ping_alternate_uncertainty_diversity : bool, optional
        Only affects reveal-one-by-one mode. When True, ping picks alternate between
        uncertainty-only (max ``_H``) and diversity-only (max reveal diversity term), starting
        with uncertainty. Compensation picks do not advance the alternation.
    granule_prior_round_robin : bool, optional
        If True (and per-label mode is active), use vote-vector granules with class priors,
        no pong compensation, and target-class round-robin searching.
    granule_prior_cap : float, optional
        Upper cap for the boosted prior of the revealed true class when updating a granule prior
        (default: 0.95).
    use_random_pong : bool, optional
        If True (default), alternate score-based turns with random-gold turns (ping-pong).
        If False, disable random turns and use score-based selection on every turn.
    pong_score : str, optional
        Pong-turn policy when ``use_random_pong`` is True.
        - ``"random"``: random pick from underrepresented gold class(es) (default).
        - otherwise: choose by score from underrepresented gold class(es) using one of
          ``{"harmonic","sum","product","uncertainty","diversity"}``.
    global_pool_round_local_diversity_after_first : bool, optional
        Only affects the round-based global-pool ping path (``use_pred_round_robin=False``
        with mismatch/per-ping restoration enabled). When True, the first ping pick in each
        ping round uses diversity relative to the accumulated selected set ``S`` (default
        behavior), while subsequent ping picks in that same round use diversity relative only
        to earlier ping picks from that round.

    Returns
    -------
    List[Tuple[str, str, float, Optional[str], Optional[str]]]
        Tuples of (file_id, text, entropy, gold_label_or_None, predicted_label_or_None).
    """
    if k <= 0 or table is None or table.empty:
        return []
    for col in (id_col, pred_col, entropy_col):
        if col not in table.columns:
            raise ValueError(f"Missing column '{col}' in table.")

    rng = np.random.default_rng(seed)

    def _default_loader(fname: str) -> str:
        if directory is None:
            raise ValueError(
                "No text_loader provided and directory=None. "
                "Pass a `text_loader` or a `directory` to read from."
            )
        p = Path(directory) / str(fname)
        if not p.exists():
            return f"[{fname} not found in {directory}]"
        try:
            return p.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            return f"[ERROR reading {fname}: {e}]"

    loader = text_loader or _default_loader

    if gold_df is not None:
        if gold_on not in gold_df.columns or gold_label_col not in gold_df.columns:
            raise ValueError(f"gold_df must contain '{gold_on}' and '{gold_label_col}'")
        _gold_names = [str(x) for x in gold_df[gold_on].tolist()]
        _gold_map = dict(zip(_gold_names, [str(x) for x in gold_df[gold_label_col].tolist()]))

        def gold_label_for(name: str) -> Optional[str]:
            s = str(name)
            match = next((g for g in _gold_names if g in s), None)
            return _gold_map.get(match) if match is not None else None
    else:
        def gold_label_for(_: str) -> Optional[str]:
            return None

    base_cols = [id_col, pred_col, entropy_col]
    if "prediction_vector" in table.columns:
        base_cols.append("prediction_vector")
    if "counts" in table.columns:
        base_cols.append("counts")
    base = table[base_cols].copy()
    base[id_col] = base[id_col].astype(str)
    base["_gold"] = base[id_col].apply(gold_label_for)

    per_label_mode = bool(per_label and gold_df is not None)
    if per_label and gold_df is None:
        per_label_mode = False

    if per_label_mode:
        base = base[base["_gold"].notna()].reset_index(drop=True)
        if base.empty:
            return []

    ent = base[entropy_col].astype(float).to_numpy()
    if ent.size == 0:
        return []
    e_min, e_max = float(np.min(ent)), float(np.max(ent))
    if e_max > e_min:
        H_hat = (ent - e_min) / (e_max - e_min)
    else:
        H_hat = np.zeros_like(ent)
    base["_H"] = H_hat

    ids = base[id_col].tolist()
    texts = [loader(fid) for fid in ids]

    _embedder = embedder
    if embedding_cache is not None:
        try:
            embs = np.stack([embedding_cache[fid] for fid in ids])
        except KeyError as exc:
            missing = exc.args[0]
            raise KeyError(f"Embedding cache missing vector for '{missing}'") from exc
    else:
        if _embedder is None:
            _embedder = build_e5_embedder(e5_model_name, device=e5_device, prefix=e5_prefix)

        try:
            embs = _embedder(texts)
        except Exception as e:
            raise RuntimeError(f"Embedding failed: {e}")

        if not isinstance(embs, np.ndarray):
            embs = np.asarray(embs)
        if embs.ndim != 2 or embs.shape[0] != len(ids):
            raise ValueError("Embedder returned unexpected shape.")

    N = len(ids)
    if N == 0:
        return []

    remaining = set(range(N))
    selected: List[int] = []
    max_sim_to_S = np.zeros(N, dtype=np.float32)

    gold_labels = base["_gold"].tolist()
    pred_labels = base[pred_col].astype(str).tolist()

    uniq_gold = sorted({g for g in gold_labels if g is not None})
    counts_gold: Dict[Optional[str], int] = {g: 0 for g in uniq_gold}

    pred_order = list(base.groupby(pred_col, sort=False).groups.keys())
    pred_pointer = 0
    last_pred_used: Optional[str] = None
    consec_pred_count = 0

    quota = k
    reveal_one_by_one_enabled = bool(reveal_one_by_one and per_label_mode and counts_gold)
    reveal_gap_threshold = max(0, int(reveal_one_by_one_gap_threshold))
    reveal_no_pong_compensation_enabled = bool(
        reveal_no_pong_compensation and reveal_one_by_one_enabled and counts_gold
    )
    reveal_max_sim_to_gold: Dict[str, np.ndarray] = {}
    reveal_pred_pointer = 0
    reveal_ping_pred_rr_enabled = bool(
        reveal_ping_pred_round_robin and reveal_one_by_one_enabled and pred_order
    )
    reveal_ping_pred_prob_update_enabled = bool(
        reveal_ping_pred_round_robin_prob_update and reveal_one_by_one_enabled and counts_gold
    )
    reveal_ping_pred_deficit_h_enabled = bool(reveal_ping_pred_deficit_h and reveal_one_by_one_enabled and counts_gold)
    reveal_ping_random_pool_enabled = bool(reveal_ping_random_pool and reveal_one_by_one_enabled and counts_gold)
    reveal_ping_alt_ud_enabled = bool(
        reveal_ping_alternate_uncertainty_diversity and reveal_one_by_one_enabled and counts_gold
    )
    granule_prior_round_robin_enabled = bool(granule_prior_round_robin and per_label_mode and counts_gold)
    reveal_ping_policy_overrides = (
        int(reveal_ping_pred_rr_enabled)
        + int(reveal_ping_pred_prob_update_enabled)
        + int(reveal_ping_pred_deficit_h_enabled)
        + int(reveal_ping_random_pool_enabled)
        + int(reveal_ping_alt_ud_enabled)
    )
    if reveal_ping_policy_overrides > 1:
        raise ValueError(
            "At most one reveal ping override may be enabled: "
            "{reveal_ping_pred_round_robin, reveal_ping_pred_round_robin_prob_update, "
            "reveal_ping_pred_deficit_h, "
            "reveal_ping_random_pool, reveal_ping_alternate_uncertainty_diversity}"
        )
    if reveal_one_by_one_enabled:
        reveal_max_sim_to_gold = {str(g): np.zeros(N, dtype=np.float32) for g in counts_gold.keys() if g is not None}

    def diversity(i: int) -> float:
        return 1.0 - float(max_sim_to_S[i])

    def score_with_mode(mode: str, h: float, d: float, *, has_anchor: bool) -> float:
        # Before the first exemplar is selected, diversity is undefined.
        # Force uncertainty-only scoring for the first pick.
        if not has_anchor:
            return h
        if mode == "harmonic":
            denom = (h + d) if (h + d) > 0 else 1e-12
            return (2.0 * h * d) / denom
        if mode == "sum":
            return float(alpha) * h + (1.0 - float(alpha)) * d
        if mode == "product":
            return h * d
        if mode == "uncertainty":
            return h
        if mode == "diversity":
            return d
        raise ValueError(
            "score modes must be one of {'harmonic','sum','product','uncertainty','diversity'}"
        )

    def score_fn(h: float, d: float) -> float:
        return score_with_mode(score, h, d, has_anchor=bool(selected))

    def update_max_sims(new_idx: int) -> None:
        v = embs[new_idx]
        sims = embs @ v
        if remaining:
            r_idx = np.fromiter(remaining, dtype=int)
            max_sim_to_S[r_idx] = np.maximum(max_sim_to_S[r_idx], sims[r_idx])
            if reveal_one_by_one_enabled:
                g_new = gold_labels[new_idx]
                if g_new is not None:
                    g_key = str(g_new)
                    if g_key in reveal_max_sim_to_gold:
                        reveal_max_sim_to_gold[g_key][r_idx] = np.maximum(
                            reveal_max_sim_to_gold[g_key][r_idx],
                            sims[r_idx],
                        )

    def quota_met(label: Optional[str]) -> bool:
        if not per_label_mode or label is None:
            return False
        return counts_gold.get(label, 0) >= quota

    def quotas_satisfied() -> bool:
        if not per_label_mode or not counts_gold:
            return len(selected) >= k
        for label, count in counts_gold.items():
            if count >= quota:
                continue
            if any(gold_labels[i] == label for i in remaining):
                return False
        return True

    def pick_random_from_gold_underrepresented() -> Optional[int]:
        if not uniq_gold:
            return None
        avail_by_g: Dict[str, List[int]] = {}
        for i in remaining:
            g = gold_labels[i]
            if g is None or quota_met(g):
                continue
            avail_by_g.setdefault(g, []).append(i)
        if not avail_by_g:
            return None
        min_count = min(counts_gold.get(g, 0) for g in avail_by_g)
        underrep = [g for g in avail_by_g if counts_gold.get(g, 0) == min_count]
        g_choice = underrep[int(rng.integers(low=0, high=len(underrep)))]
        pool = avail_by_g[g_choice]
        return int(rng.choice(pool)) if pool else None

    def next_pred_bucket_indices() -> List[int]:
        nonlocal pred_pointer, last_pred_used, consec_pred_count
        candidates: List[int] = []
        if use_pred_round_robin and pred_order:
            for _ in range(len(pred_order)):
                p = pred_order[pred_pointer]
                pred_pointer = (pred_pointer + 1) % len(pred_order)
                if last_pred_used == p and consec_pred_count >= max_consecutive_pred:
                    continue
                bucket = [i for i in remaining if pred_labels[i] == p]
                if per_label_mode:
                    bucket = [i for i in bucket if not quota_met(gold_labels[i])]
                if bucket:
                    candidates.extend(bucket)
                    break
        if not candidates:
            candidates = [i for i in remaining]
            if per_label_mode:
                candidates = [i for i in candidates if not quota_met(gold_labels[i])]
        return candidates

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

    results: List[Tuple[str, str, float, Optional[str], Optional[str]]] = []
    selection_events: List[Dict[str, Any]] = []
    ping_selected = 0
    random_selected = 0
    mismatch_ping = 0
    restore_rounds = 0
    restore_warmup = max(0, int(pong_restore_warmup_pings))

    def _record_pick(i: int, *, mode: str, round_idx: Optional[int], turn_idx: Optional[int]) -> None:
        nonlocal ping_selected, random_selected, mismatch_ping
        fid = ids[i]
        gold = gold_labels[i]
        pred = pred_labels[i]
        pred_gold_match: Optional[bool] = None
        if gold is not None:
            pred_gold_match = (_canon_label(pred) == _canon_label(gold))

        if mode.startswith("ping"):
            ping_selected += 1
            if pred_gold_match is False:
                mismatch_ping += 1
        if "random" in mode:
            random_selected += 1

        selection_events.append(
            {
                "select_order": len(selection_events) + 1,
                "round_idx": round_idx,
                "turn_idx": turn_idx,
                "mode": mode,
                "selected_file": fid,
                "pred_label": pred,
                "gold_label": gold,
                "pred_gold_match": pred_gold_match,
                "entropy": float(base[entropy_col].iat[i]),
            }
        )

    def _run_random_restore(round_idx: int) -> bool:
        nonlocal restore_rounds
        if not counts_gold:
            return False

        made_any = False
        target_count = min(quota, max(counts_gold.values()))
        for g_label in sorted(counts_gold.keys(), key=_gold_sort_key):
            if quotas_satisfied():
                break
            need = max(0, target_count - counts_gold.get(g_label, 0))
            if need <= 0:
                continue
            pool = [
                i for i in remaining
                if gold_labels[i] == g_label and not quota_met(gold_labels[i])
            ]
            if not pool:
                continue
            n_pick = min(need, len(pool))
            chosen = rng.choice(pool, size=n_pick, replace=False)
            chosen_idxs = [int(x) for x in np.atleast_1d(chosen).tolist()]
            for pick in chosen_idxs:
                if pick not in remaining:
                    continue
                remaining.remove(pick)
                selected.append(pick)
                update_max_sims(pick)
                g = gold_labels[pick]
                if g in counts_gold:
                    counts_gold[g] += 1
                _record_pick(pick, mode="pong_random_restore", round_idx=round_idx, turn_idx=None)
                fid = ids[pick]
                results.append((fid, texts[pick], float(base[entropy_col].iat[pick]), g, pred_labels[pick]))
                made_any = True
                if quotas_satisfied():
                    break

        if made_any:
            restore_rounds += 1
        return made_any

    def _reveal_ping_diversity(i: int) -> float:
        if not selected:
            return 1.0
        if not reveal_one_by_one_enabled or not counts_gold:
            return diversity(i)
        count_vals = list(counts_gold.values())
        if not count_vals:
            return diversity(i)
        max_count = max(count_vals)
        min_count = min(count_vals)
        if max_count <= min_count:
            return diversity(i)

        overrep_labels = [str(g) for g, c in counts_gold.items() if g is not None and c == max_count]
        if not overrep_labels:
            return diversity(i)

        max_sim = 0.0
        found = False
        for g in overrep_labels:
            arr = reveal_max_sim_to_gold.get(g)
            if arr is None:
                continue
            max_sim = max(max_sim, float(arr[i]))
            found = True
        return (1.0 - max_sim) if found else diversity(i)

    def _pick_random_reveal_compensation(
        avail_by_g: Dict[str, List[int]],
        deficits: Dict[str, int],
    ) -> Optional[int]:
        if not avail_by_g:
            return None
        max_def = max(int(deficits.get(g, 0)) for g in avail_by_g.keys())
        largest_deficit_labels = [
            g for g in sorted(avail_by_g.keys(), key=_gold_sort_key)
            if int(deficits.get(g, 0)) == max_def
        ]
        if not largest_deficit_labels:
            return None
        g_choice = largest_deficit_labels[int(rng.integers(low=0, high=len(largest_deficit_labels)))]
        pool = avail_by_g.get(g_choice, [])
        if not pool:
            return None
        return int(rng.choice(pool))

    def _pick_reveal_ping_by_pred_deficit_h(
        candidates: List[int],
        deficits: Dict[str, int],
    ) -> Optional[int]:
        if not candidates or not deficits:
            return None

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

        for deficit_label in deficit_labels:
            target_canon = _canon_label(deficit_label)
            if target_canon is None:
                continue
            bucket = [i for i in candidates if _canon_label(pred_labels[i]) == target_canon]
            if not bucket:
                continue

            best_i: Optional[int] = None
            best_h = -1.0
            for i in bucket:
                h = float(base["_H"].iat[i])
                if (h > best_h) or (np.isclose(h, best_h) and (best_i is None or i < best_i)):
                    best_i, best_h = i, h
            if best_i is not None:
                return int(best_i)

        return None

    def _pick_reveal_pred_round_robin(candidates: List[int]) -> Optional[int]:
        nonlocal reveal_pred_pointer
        if not candidates or not pred_order:
            return None

        candidate_set = set(candidates)
        for _ in range(len(pred_order)):
            pred_bucket = pred_order[reveal_pred_pointer]
            reveal_pred_pointer = (reveal_pred_pointer + 1) % len(pred_order)
            bucket = sorted(i for i in candidate_set if pred_labels[i] == pred_bucket)
            if bucket:
                # Match notebook regime: no U/D scoring within bucket.
                return int(bucket[0])
        return None

    def _target_class_index(label: Optional[str], class_order: List[str]) -> Optional[int]:
        if label is None:
            return None
        canon = _canon_label(label)
        for idx, g in enumerate(class_order):
            if _canon_label(g) == canon:
                return idx
        return None

    def _pick_reveal_ping_component(
        candidates: List[int],
        *,
        component: str,
    ) -> Optional[int]:
        if not candidates:
            return None
        if component not in {"uncertainty", "diversity"}:
            raise ValueError("component must be one of {'uncertainty','diversity'}")

        best_i: Optional[int] = None
        best_primary = -1.0
        best_secondary = -1.0
        for i in candidates:
            h = float(base["_H"].iat[i])
            d = _reveal_ping_diversity(i)
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

    prob_update_class_order: List[str] = []
    prob_update_rr_pointer = 0
    prob_update_class_probs: Optional[np.ndarray] = None
    prob_update_sim_matrix: Optional[np.ndarray] = None
    prob_update_sim_mean = 0.0

    if reveal_ping_pred_prob_update_enabled:
        prob_update_class_order = [g for g in sorted(counts_gold.keys(), key=_gold_sort_key) if g is not None]
        if not prob_update_class_order:
            reveal_ping_pred_prob_update_enabled = False
        else:
            n_classes = len(prob_update_class_order)
            prob_update_class_probs = np.zeros((N, n_classes), dtype=np.float64)
            counts_values = (
                base["counts"].tolist()
                if "counts" in base.columns
                else [None for _ in range(N)]
            )
            uniform_row = np.full(n_classes, 1.0 / float(n_classes), dtype=np.float64)

            for i in range(N):
                row = prob_update_class_probs[i]
                counts_obj = counts_values[i]
                if isinstance(counts_obj, dict):
                    for key, val in counts_obj.items():
                        idx = _target_class_index(str(key), prob_update_class_order)
                        if idx is None:
                            continue
                        try:
                            w = max(0.0, float(val))
                        except Exception:
                            w = 0.0
                        row[idx] += w

                if float(np.sum(row)) <= 0.0:
                    idx = _target_class_index(pred_labels[i], prob_update_class_order)
                    if idx is not None:
                        row[idx] = 1.0

                row_sum = float(np.sum(row))
                if row_sum > 0.0 and np.isfinite(row_sum):
                    prob_update_class_probs[i] = row / row_sum
                else:
                    prob_update_class_probs[i] = uniform_row

            prob_update_sim_matrix = (embs @ embs.T).astype(np.float32, copy=False)
            if N > 1:
                tri = prob_update_sim_matrix[np.triu_indices(N, k=1)]
                prob_update_sim_mean = float(np.mean(tri)) if tri.size > 0 else 0.0
            else:
                prob_update_sim_mean = 0.0

    # New granule-prior no-pong round-robin mode.
    if granule_prior_round_robin_enabled:
        class_cycle = [g for g in sorted(counts_gold.keys(), key=_gold_sort_key) if g is not None]
        class_canons = []
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
                base["_granule_key"] = base["prediction_vector"].fillna("__MISSING_VECTOR__").astype(str)
            elif "counts" in base.columns:
                base["_granule_key"] = base["counts"].map(
                    lambda x: str(sorted((str(k), int(v)) for k, v in x.items())) if isinstance(x, dict) else "__NO_COUNTS__"
                )
            else:
                base["_granule_key"] = base[pred_col].astype(str)

            granule_keys = base["_granule_key"].astype(str).tolist()
            granule_to_all_idxs: Dict[str, List[int]] = {}
            for i, gk in enumerate(granule_keys):
                granule_to_all_idxs.setdefault(gk, []).append(i)
            remaining_by_granule: Dict[str, set[int]] = {
                gk: set(idxs) for gk, idxs in granule_to_all_idxs.items()
            }

            granule_prior: Dict[str, Dict[str, float]] = {}
            granule_entropy: Dict[str, float] = {}
            granule_avg_sim: Dict[str, float] = {}

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

            turn = 1
            rr_pointer = 0
            prior_cap = float(min(max(granule_prior_cap, 0.0), 1.0))

            while remaining and not quotas_satisfied():
                active_classes = [g for g in class_cycle if not quota_met(g)]
                if not active_classes:
                    break

                target_class: Optional[str] = None
                for _ in range(len(class_cycle)):
                    cand = class_cycle[rr_pointer]
                    rr_pointer = (rr_pointer + 1) % len(class_cycle)
                    if cand in active_classes:
                        target_class = cand
                        break
                if target_class is None:
                    break
                target_canon = _canon_label(target_class)
                if target_canon is None:
                    target_canon = str(target_class)

                available_granules = [gk for gk, idxs in remaining_by_granule.items() if idxs]
                pick: Optional[int] = None
                mode = "ping_granule_prior_rr"

                if available_granules:
                    non_singletons = [gk for gk in available_granules if len(remaining_by_granule[gk]) > 1]
                    candidate_granules = non_singletons if non_singletons else available_granules
                    max_size = max(len(remaining_by_granule[gk]) for gk in candidate_granules) if candidate_granules else 0

                    granule_scores: List[float] = []
                    for gk in candidate_granules:
                        p_t = float(granule_prior[gk].get(target_canon, 0.0))
                        h_g = float(granule_entropy.get(gk, 0.0))
                        size_norm = (float(len(remaining_by_granule[gk])) / float(max_size)) if max_size > 0 else 0.0
                        s_g = _harmonic3(p_t, h_g, size_norm)
                        granule_scores.append(max(0.0, float(s_g)))

                    if candidate_granules:
                        probs = np.asarray(granule_scores, dtype=float)
                        total = float(np.sum(probs))
                        if total > 0.0:
                            probs = probs / total
                        else:
                            probs = np.full(len(candidate_granules), 1.0 / float(len(candidate_granules)), dtype=float)
                        chosen_granule = str(rng.choice(candidate_granules, p=probs))
                        granule_pool = sorted(remaining_by_granule[chosen_granule])
                        if granule_pool:
                            pick = int(rng.choice(granule_pool))
                if pick is None:
                    # Fallback only when no granule candidate could be used.
                    pool = sorted(list(remaining))
                    if not pool:
                        break
                    pick = int(rng.choice(pool))
                    mode = "ping_granule_fallback_random"

                if pick not in remaining:
                    break

                gk_pick = granule_keys[pick]
                remaining.remove(pick)
                if gk_pick in remaining_by_granule:
                    remaining_by_granule[gk_pick].discard(pick)

                g = gold_labels[pick]
                g_canon = _canon_label(g) if g is not None else None

                # Update the selected granule prior after revealing gold label.
                if g_canon is not None and gk_pick in granule_prior and g_canon in granule_prior[gk_pick]:
                    all_idx_arr = np.asarray(granule_to_all_idxs[gk_pick], dtype=int)
                    if all_idx_arr.size > 0:
                        sims_to_pick = embs[all_idx_arr] @ embs[pick]
                        frac = float(np.mean(sims_to_pick > float(granule_avg_sim.get(gk_pick, 1.0))))
                    else:
                        frac = 0.0
                    p = granule_prior[gk_pick]
                    old_true = float(p.get(g_canon, 0.0))
                    new_true = min(prior_cap, old_true + frac)
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

                accepted = bool(g in counts_gold and not quota_met(g))
                event_mode = mode if accepted else f"{mode}_rejected_full"
                _record_pick(pick, mode=event_mode, round_idx=None, turn_idx=turn)
                if accepted:
                    selected.append(pick)
                    update_max_sims(pick)
                    counts_gold[g] += 1
                    fid = ids[pick]
                    results.append((fid, texts[pick], float(base[entropy_col].iat[pick]), g, pred_labels[pick]))
                turn += 1

    # New round-based conditional pong: run random restoration only when a ping round
    # produced at least one pred!=gold pick.
    elif reveal_one_by_one_enabled:
        turn = 1
        reveal_budget_total = min(int(quota) * len(counts_gold), N)
        reveal_ping_component_count = 0

        while remaining and (
            not quotas_satisfied() if reveal_no_pong_compensation_enabled else len(selected) < reveal_budget_total
        ):
            avail_by_g: Dict[str, List[int]] = {}
            for i in remaining:
                g = gold_labels[i]
                if g is None:
                    continue
                avail_by_g.setdefault(str(g), []).append(i)

            deficits: Dict[str, int] = {}
            for g, count in counts_gold.items():
                if g is None:
                    continue
                deficits[str(g)] = max(0, int(quota) - int(count))

            deficit_avail = {
                g: pool for g, pool in avail_by_g.items()
                if int(deficits.get(g, 0)) > 0 and len(pool) > 0
            }
            count_vals = list(counts_gold.values())
            gap_now = (max(count_vals) - min(count_vals)) if count_vals else 0
            remaining_slots = max(0, reveal_budget_total - len(selected))
            deficit_total = sum(int(v) for v in deficits.values())
            nondeficit_avail_exists = any(
                int(deficits.get(g, 0)) <= 0 and len(pool) > 0
                for g, pool in avail_by_g.items()
            )

            force_compensation = bool(deficit_avail) and nondeficit_avail_exists and (deficit_total >= remaining_slots)
            early_compensation = bool(deficit_avail) and (gap_now >= reveal_gap_threshold > 0)

            event_mode: Optional[str] = None
            pick: Optional[int] = None
            expected_target_idx: Optional[int] = None

            if (
                not reveal_no_pong_compensation_enabled
                and (force_compensation or early_compensation)
            ):
                pick = _pick_random_reveal_compensation(deficit_avail, deficits)
                if pick is not None:
                    event_mode = "pong_random_compensate"

            if pick is None:
                # Ping from the unlabeled pool: no gold-based prefiltering. Gold is only revealed after selection.
                candidates = list(remaining)
                if not candidates:
                    break

                if reveal_ping_pred_prob_update_enabled and prob_update_class_probs is not None:
                    active_classes = [
                        g for g in prob_update_class_order
                        if not quota_met(g)
                    ]
                    target_label: Optional[str] = None
                    for _ in range(len(prob_update_class_order)):
                        cand = prob_update_class_order[prob_update_rr_pointer]
                        prob_update_rr_pointer = (prob_update_rr_pointer + 1) % len(prob_update_class_order)
                        if cand in active_classes:
                            target_label = cand
                            break

                    if target_label is not None:
                        target_idx = _target_class_index(target_label, prob_update_class_order)
                        cand_arr = np.asarray(candidates, dtype=int)
                        if target_idx is not None and cand_arr.size > 0:
                            weights = np.asarray(
                                prob_update_class_probs[cand_arr, target_idx],
                                dtype=np.float64,
                            )
                            weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
                            total = float(np.sum(weights))
                            if total > 0.0:
                                probs = weights / total
                                pick = int(rng.choice(cand_arr, p=probs))
                            else:
                                pick = int(rng.choice(candidates))
                            expected_target_idx = int(target_idx)
                            event_mode = "ping_reveal_pred_round_robin_prob_update"

                if pick is None and reveal_ping_pred_rr_enabled:
                    pick = _pick_reveal_pred_round_robin(candidates)
                    if pick is not None:
                        event_mode = "ping_reveal_pred_round_robin"

                if pick is None and reveal_ping_pred_deficit_h_enabled:
                    # Target the largest current gold deficit via predicted label, then pick max precomputed entropy.
                    pick = _pick_reveal_ping_by_pred_deficit_h(candidates, deficits)
                    if pick is not None:
                        event_mode = "ping_reveal_pred_deficit_h"

                if pick is None and reveal_ping_random_pool_enabled:
                    pick = int(rng.choice(candidates))
                    event_mode = "ping_reveal_random_pool"

                if pick is None and reveal_ping_alt_ud_enabled:
                    component = "uncertainty" if (reveal_ping_component_count % 2 == 0) else "diversity"
                    pick = _pick_reveal_ping_component(candidates, component=component)
                    if pick is not None:
                        event_mode = (
                            "ping_reveal_uncertainty"
                            if component == "uncertainty"
                            else "ping_reveal_diversity"
                        )

                if pick is None:
                    best_i: Optional[int] = None
                    best_s = -1.0
                    best_h = -1.0
                    for i in candidates:
                        h = float(base["_H"].iat[i])
                        d = _reveal_ping_diversity(i)
                        s = score_fn(h, d)
                        if (s > best_s) or (np.isclose(s, best_s) and h > best_h):
                            best_i, best_s, best_h = i, s, h

                    pick = int(rng.choice(candidates)) if best_i is None else best_i
                    event_mode = "ping_reveal"

            if pick not in remaining:
                break

            remaining.remove(pick)
            g = gold_labels[pick]

            if (
                reveal_ping_pred_prob_update_enabled
                and expected_target_idx is not None
                and prob_update_class_probs is not None
                and prob_update_sim_matrix is not None
            ):
                revealed_idx = _target_class_index(g, prob_update_class_order)
                if revealed_idx is not None and int(revealed_idx) != int(expected_target_idx):
                    if remaining:
                        r_idx = np.fromiter(remaining, dtype=int)
                        sims = prob_update_sim_matrix[pick, r_idx]
                        affected = r_idx[sims > float(prob_update_sim_mean)]
                        if affected.size > 0:
                            moved = prob_update_class_probs[affected, expected_target_idx].copy()
                            prob_update_class_probs[affected, expected_target_idx] = 0.0
                            prob_update_class_probs[affected, revealed_idx] += moved

            accepted = True
            if reveal_no_pong_compensation_enabled:
                accepted = bool(g in counts_gold and not quota_met(g))
            mode = str(event_mode) if accepted else f"{event_mode}_rejected_full"
            _record_pick(pick, mode=mode, round_idx=None, turn_idx=turn)
            if accepted:
                selected.append(pick)
                update_max_sims(pick)
                if g in counts_gold:
                    counts_gold[g] += 1
                fid = ids[pick]
                results.append((fid, texts[pick], float(base[entropy_col].iat[pick]), g, pred_labels[pick]))
            if reveal_ping_alt_ud_enabled and str(event_mode).startswith("ping"):
                reveal_ping_component_count += 1

            turn += 1

    elif use_random_pong and per_label_mode and (pong_restore_on_mismatch_only or pong_restore_after_each_ping):
        round_idx = 0
        while remaining and not quotas_satisfied():
            round_idx += 1
            round_has_mismatch = False
            any_ping_in_round = False

            if use_pred_round_robin:
                # One ping candidate per predicted-majority bucket.
                for pred_bucket in pred_order:
                    if not remaining or quotas_satisfied():
                        break
                    bucket = [i for i in remaining if pred_labels[i] == pred_bucket]
                    if per_label_mode:
                        bucket = [i for i in bucket if gold_labels[i] is not None and not quota_met(gold_labels[i])]
                    if not bucket:
                        continue

                    best_i: Optional[int] = None
                    best_s = -1.0
                    for i in bucket:
                        g = gold_labels[i]
                        if per_label_mode and (g is None or quota_met(g)):
                            continue
                        h = float(base["_H"].iat[i])
                        d = diversity(i)
                        s = score_fn(h, d)
                        if s > best_s:
                            best_i, best_s = i, s
                    if best_i is None:
                        continue

                    any_ping_in_round = True
                    remaining.remove(best_i)
                    selected.append(best_i)
                    update_max_sims(best_i)
                    g = gold_labels[best_i]
                    if g in counts_gold:
                        counts_gold[g] += 1
                    _record_pick(best_i, mode="ping", round_idx=round_idx, turn_idx=None)
                    mismatch_this_pick = (_canon_label(pred_labels[best_i]) != _canon_label(g))
                    if mismatch_this_pick:
                        round_has_mismatch = True

                    fid = ids[best_i]
                    results.append((fid, texts[best_i], float(base[entropy_col].iat[best_i]), g, pred_labels[best_i]))

                    if (
                        pong_restore_after_each_ping
                        and ping_selected >= restore_warmup
                        and (not pong_restore_on_mismatch_only or mismatch_this_pick)
                    ):
                        _run_random_restore(round_idx)
                        if quotas_satisfied():
                            break
            else:
                # Global-pool ping: perform one round of argmax score picks across ALL
                # remaining candidates, with round budget equal to number of predicted classes.
                round_budget = max(1, len(pred_order)) if pred_order else 1
                picks_done = 0
                round_ping_max_sim = np.zeros(N, dtype=np.float32)
                while picks_done < round_budget and remaining and not quotas_satisfied():
                    candidates = [i for i in remaining]
                    if per_label_mode:
                        candidates = [
                            i for i in candidates
                            if gold_labels[i] is not None and not quota_met(gold_labels[i])
                        ]
                    if not candidates:
                        break

                    best_i: Optional[int] = None
                    best_s = -1.0
                    for i in candidates:
                        g = gold_labels[i]
                        if per_label_mode and (g is None or quota_met(g)):
                            continue
                        h = float(base["_H"].iat[i])
                        if (
                            global_pool_round_local_diversity_after_first
                            and picks_done > 0
                        ):
                            d = 1.0 - float(round_ping_max_sim[i])
                        else:
                            d = diversity(i)
                        s = score_fn(h, d)
                        if s > best_s:
                            best_i, best_s = i, s
                    if best_i is None:
                        break

                    any_ping_in_round = True
                    picks_done += 1
                    remaining.remove(best_i)
                    selected.append(best_i)
                    update_max_sims(best_i)
                    if global_pool_round_local_diversity_after_first:
                        v_round = embs[best_i]
                        sims_round = embs @ v_round
                        if remaining:
                            r_idx_round = np.fromiter(remaining, dtype=int)
                            round_ping_max_sim[r_idx_round] = np.maximum(
                                round_ping_max_sim[r_idx_round],
                                sims_round[r_idx_round],
                            )
                    g = gold_labels[best_i]
                    if g in counts_gold:
                        counts_gold[g] += 1
                    _record_pick(best_i, mode="ping_global", round_idx=round_idx, turn_idx=None)
                    mismatch_this_pick = (_canon_label(pred_labels[best_i]) != _canon_label(g))
                    if mismatch_this_pick:
                        round_has_mismatch = True

                    fid = ids[best_i]
                    results.append((fid, texts[best_i], float(base[entropy_col].iat[best_i]), g, pred_labels[best_i]))

                    if (
                        pong_restore_after_each_ping
                        and ping_selected >= restore_warmup
                        and (not pong_restore_on_mismatch_only or mismatch_this_pick)
                    ):
                        _run_random_restore(round_idx)
                        if quotas_satisfied():
                            break

            if (
                not use_random_pong
                or not any_ping_in_round
                or pong_restore_after_each_ping
                or not round_has_mismatch
                or quotas_satisfied()
            ):
                continue

            # Random restoration after mismatch: rebalance each gold class up to current max count
            # (bounded by quota). This keeps classes close after each ping/pong iteration.
            _run_random_restore(round_idx)

    else:
        turn = 1
        while remaining and not quotas_satisfied():
            ping_turn = (turn % 2 == 1) or (not use_random_pong)
            if ping_turn:
                cand_idxs = next_pred_bucket_indices()
                if not cand_idxs:
                    cand_idxs = [i for i in remaining if not quota_met(gold_labels[i])]
                    if not cand_idxs:
                        cand_idxs = list(remaining)
                best_i: Optional[int] = None
                best_s = -1.0
                best_pred: Optional[str] = None
                for i in cand_idxs:
                    g = gold_labels[i]
                    if per_label_mode and (g is None or quota_met(g)):
                        continue
                    h = float(base["_H"].iat[i])
                    d = diversity(i)
                    s = score_fn(h, d)
                    if s > best_s:
                        best_i, best_s, best_pred = i, s, pred_labels[i]
                if best_i is None:
                    fallback = [i for i in cand_idxs if not quota_met(gold_labels[i])]
                    if per_label_mode:
                        fallback = [i for i in fallback if gold_labels[i] is not None]
                    if not fallback:
                        fallback = [i for i in remaining if not quota_met(gold_labels[i])]
                        if per_label_mode:
                            fallback = [i for i in fallback if gold_labels[i] is not None]
                    if not fallback:
                        fallback = list(remaining)
                    if not fallback:
                        break
                    best_i = int(rng.choice(fallback))
                    best_pred = pred_labels[best_i]

                remaining.remove(best_i)
                selected.append(best_i)
                update_max_sims(best_i)
                g = gold_labels[best_i]
                if g in counts_gold:
                    counts_gold[g] += 1
                prev_pred = last_pred_used
                last_pred_used = best_pred
                consec_pred_count = (consec_pred_count + 1) if prev_pred == best_pred else 1
                _record_pick(best_i, mode="ping", round_idx=None, turn_idx=turn)

                fid = ids[best_i]
                results.append((fid, texts[best_i], float(base[entropy_col].iat[best_i]), g, pred_labels[best_i]))

            else:
                pick: Optional[int] = None
                pong_mode = str(pong_score).strip().lower()

                if pong_mode == "random":
                    pick = pick_random_from_gold_underrepresented()
                    if pick is None:
                        with_gold = [i for i in remaining if gold_labels[i] is not None and not quota_met(gold_labels[i])]
                        pool = with_gold if with_gold else [i for i in remaining if not quota_met(gold_labels[i])]
                        if not pool:
                            pool = list(remaining)
                        if not pool:
                            break
                        pick = int(rng.choice(pool))
                else:
                    # Score-based pong: choose from underrepresented gold class(es) when possible.
                    pool: List[int] = []
                    avail_by_g: Dict[str, List[int]] = {}
                    for i in remaining:
                        g = gold_labels[i]
                        if g is None or quota_met(g):
                            continue
                        avail_by_g.setdefault(g, []).append(i)
                    if avail_by_g:
                        min_count = min(counts_gold.get(g, 0) for g in avail_by_g)
                        underrep = [g for g in avail_by_g if counts_gold.get(g, 0) == min_count]
                        for g in underrep:
                            pool.extend(avail_by_g[g])
                    if not pool:
                        pool = [i for i in remaining if not quota_met(gold_labels[i])]
                    if not pool:
                        pool = list(remaining)
                    if not pool:
                        break

                    best_i: Optional[int] = None
                    best_s = -1.0
                    for i in pool:
                        g = gold_labels[i]
                        if per_label_mode and (g is None or quota_met(g)):
                            continue
                        h = float(base["_H"].iat[i])
                        d = diversity(i)
                        s = score_with_mode(pong_mode, h, d, has_anchor=bool(selected))
                        if s > best_s:
                            best_i, best_s = i, s
                    pick = int(rng.choice(pool)) if best_i is None else best_i

                remaining.remove(pick)
                selected.append(pick)
                update_max_sims(pick)
                g = gold_labels[pick]
                if g in counts_gold:
                    counts_gold[g] += 1
                last_pred_used = None
                consec_pred_count = 0
                pong_mode = str(pong_score).strip().lower()
                event_mode = "pong_random" if pong_mode == "random" else f"pong_{pong_mode}"
                _record_pick(pick, mode=event_mode, round_idx=None, turn_idx=turn)

                fid = ids[pick]
                results.append((fid, texts[pick], float(base[entropy_col].iat[pick]), g, pred_labels[pick]))

            turn += 1

    if per_label_mode and results:
        rank_map = {
            "early": 0,
            "early pregnancy": 0,
            "early active pregnancy": 0,
            "late": 1,
            "late pregnancy": 1,
            "late active pregnancy": 1,
            "unrelated": 2,
            "unrelated or no current pregnancy": 2,
            "no current pregnancy": 2,
            "no active pregnancy": 2,
            "no": 2,
        }
        ordered = {0: [], 1: [], 2: []}
        tail: List[Tuple[str, str, float, Optional[str], Optional[str]]] = []
        for item in results:
            gold = item[3] if len(item) >= 4 else None
            if gold is None:
                tail.append(item)
                continue
            key = str(gold).strip().lower()
            rank = rank_map.get(key)
            if rank is None:
                tail.append(item)
            else:
                ordered[rank].append(item)
        results = ordered[0] + ordered[1] + ordered[2] + tail

    if embedding_cache is None and _embedder is not None:
        try:
            if hasattr(_embedder, "__self__") and hasattr(_embedder.__self__, "model"):
                del _embedder.__self__.model
            elif hasattr(_embedder, "model"):
                del _embedder.model
        except Exception:
            pass
        try:
            import gc, torch  # type: ignore
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    if stats_out is not None:
        total = len(selection_events)
        stats_out.clear()
        stats_out.update(
            {
                "summary": {
                    "total_selected": total,
                    "ping_selected": ping_selected,
                    "random_selected": random_selected,
                    "ping_fraction": (float(ping_selected) / total) if total > 0 else 0.0,
                    "random_fraction": (float(random_selected) / total) if total > 0 else 0.0,
                    "mismatch_ping_selected": mismatch_ping,
                    "restore_rounds": restore_rounds,
                    "pong_restore_after_each_ping": bool(pong_restore_after_each_ping),
                    "pong_restore_warmup_pings": int(restore_warmup),
                    "per_label_mode": bool(per_label_mode),
                    "quota": int(quota),
                    "pong_restore_on_mismatch_only": bool(pong_restore_on_mismatch_only),
                    "reveal_one_by_one": bool(reveal_one_by_one_enabled),
                    "reveal_one_by_one_gap_threshold": int(reveal_gap_threshold),
                    "reveal_no_pong_compensation": bool(reveal_no_pong_compensation_enabled),
                    "reveal_ping_pred_round_robin": bool(reveal_ping_pred_rr_enabled),
                    "reveal_ping_pred_round_robin_prob_update": bool(reveal_ping_pred_prob_update_enabled),
                    "reveal_ping_pred_deficit_h": bool(reveal_ping_pred_deficit_h_enabled),
                    "reveal_ping_random_pool": bool(reveal_ping_random_pool_enabled),
                    "reveal_ping_alternate_uncertainty_diversity": bool(reveal_ping_alt_ud_enabled),
                    "granule_prior_round_robin": bool(granule_prior_round_robin_enabled),
                    "granule_prior_cap": float(min(max(granule_prior_cap, 0.0), 1.0)),
                },
                "events": selection_events,
            }
        )

    return results


from typing import List, Tuple, Optional, Dict, Union

def make_icl_prompt_from_pingpong(
    results: List[Union[
        Tuple[str, str, float, Optional[str], Optional[str]],  # (id, text, ent, gold, pred)
        Tuple[str, str, float, Optional[str]],                 # (id, text, ent, gold)
    ]],
    *,
    instruction: str = (
        "You are a medical assistant specialized in obstetric ultrasound.\n"
        "Classify the pregnancy status mentioned in the following report as one of:\n"
        "- early active pregnancy\n"
        "- late active pregnancy\n"
        "- no active pregnancy.\n"
        "Only output the label, nothing else.\n"
    ),
    input_tag: str = "Report",
    output_tag: str = "Label",
    truncate_chars: Optional[int] = None,
    drop_if_no_gold: bool = True,   # ensure ICL samples provide gold labels
    add_query_placeholder: bool = True,
    label_map: Optional[Dict[str, str]] = None,  # canonical gold -> rendered label
    include_rationales: bool = False,
    rationale_tag: str = "Rationale",
    rationale_lookup: Optional[Dict[str, str]] = None,
) -> str:
    """
    Build a few-shot prompt from ping-pong selection results (global-k).
    Uses ONLY gold labels for the ICL examples (predicted labels are ignored).

    Parameters
    ----------
    results : list of tuples
        Output of select_icls_pingpong: [(id, text, entropy, gold, pred)].
        4-tuple form (without pred) is also accepted.
    drop_if_no_gold : bool
        If True, skip any example that lacks a gold label.
    include_rationales : bool
        If True, append a rationale line for each exemplar using ``rationale_lookup``.

    Returns
    -------
    str : ready-to-use prompt text
    """

    # Default rendering for gold labels (can be overridden)
    if label_map is None:
        label_map = {
            "early": "early active pregnancy",
            "early pregnancy": "early active pregnancy",
            "early active pregnancy": "early active pregnancy",
            "late": "late active pregnancy",
            "late pregnancy": "late active pregnancy",
            "late active pregnancy": "late active pregnancy",
            "unrelated": "no active pregnancy",
            "unrelated or no current pregnancy": "no active pregnancy",
            "no current pregnancy": "no active pregnancy",
            "no active pregnancy": "no active pregnancy",
            "no": "no active pregnancy",
        }

    def _canon_gold(g: Optional[str]) -> Optional[str]:
        if g is None:
            return None
        k = str(g).strip().lower().rstrip(".")
        # normalize common variants
        if k in {"early", "early pregnancy", "early active pregnancy"}:
            k = "early"
        elif k in {"late", "late pregnancy", "late active pregnancy"}:
            k = "late"
        elif k in {
            "unrelated",
            "no current pregnancy",
            "unrelated or no current pregnancy",
            "no active pregnancy",
            "no",
            "none",
        }:
            k = "unrelated"
        # map to rendered string if available
        return label_map.get(k, k)

    lines: List[str] = [instruction.strip(), ""]

    for item in results:
        # accept 4- or 5-tuples
        if len(item) == 5:
            fid, text, ent, gold, pred = item
        elif len(item) == 4:
            fid, text, ent, gold = item
            pred = None
        else:
            raise ValueError("Each item must be (id, text, entropy, gold[, pred]).")

        gold_out = _canon_gold(gold)
        if gold_out is None and drop_if_no_gold:
            continue

        snippet = text if (truncate_chars is None or truncate_chars <= 0) else text[:truncate_chars]
        lines.append(f"{input_tag}:")
        lines.append(str(snippet).strip())
        lines.append(f"{output_tag}: {gold_out if gold_out is not None else 'UNKNOWN'}")
        if include_rationales:
            rationale = None
            if rationale_lookup is not None:
                rationale = rationale_lookup.get(str(fid))
            if rationale is not None and str(rationale).strip():
                lines.append(f"{rationale_tag}: {str(rationale).strip()}")
        lines.append("")  # blank line between examples

    if add_query_placeholder:
        lines.append("### Now classify")
        lines.append(f"{input_tag}: {{report_text}}")
        lines.append(f"{output_tag}:")

    return "\n".join(lines).strip()
