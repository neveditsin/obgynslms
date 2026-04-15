"""Utilities for leave-one-out (LOO) exemplar selection.

The goal is to remove test-item leakage during in-context learning (ICL)
selection by recomputing prompts on a per-document basis with the target
item held out.  The helpers here cache heavy computations (vote entropy,
embeddings, prompt strings) so that running LOO across many models only
pays the combinatorial cost once per document.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .data_loader import load_documents
from .evaluation_utils import evaluate_by_model
from .eval import evaluate_with_significance
from .label_utils import infer_label
from .model_runner import run_zero_shot_classification
from .vote_entropy import (
    build_e5_embedder,
    cbed_select_by_label,
    compute_vote_entropy,
    make_icl_prompt_from_pingpong,
    make_icl_prompt_from_results,
    randomk_by_gold_label,
    select_icls_pingpong,
    topk_by_top_label,
)

DEFAULT_QUERY_TEMPLATE = (
    "NOW YOUR TURN: \n"
    "Report:\n\"{report_text}\"\n\n"
    "Label: "
)

PING_SELECTIONS = {"ping", "ping_ra", "cross_ping", "cross_ping_ra"}
PING_RATIONALE_SELECTIONS = {"ping_ra", "cross_ping_ra"}
RANDOM_SELECTIONS = {"random", "random_ra"}


@dataclass
class LOOResources:
    """Bundle of pre-computed artefacts reused across LOO prompts."""

    vote_table: pd.DataFrame
    text_lookup: Dict[str, str]
    embedding_cache: Optional[Dict[str, np.ndarray]] = None
    gold_df: Optional[pd.DataFrame] = None
    id_col: str = "file"
    pred_col: str = "top_label"
    entropy_col: str = "vote_entropy"

    def ensure_id(self, doc: Dict[str, Any]) -> str:
        if self.id_col in doc:
            return str(doc[self.id_col])
        raise KeyError(f"Document dict missing '{self.id_col}' key: {doc.keys()}")

    def ensure_text(self, doc_id: str, doc: Dict[str, Any], text_key: str = "text") -> str:
        if doc_id in self.text_lookup:
            return self.text_lookup[doc_id]
        if text_key in doc and isinstance(doc[text_key], str):
            return doc[text_key]
        raise KeyError(f"No text available for document '{doc_id}'")


def prepare_loo_resources(
    predictions: pd.DataFrame,
    *,
    docs: Optional[Sequence[Dict[str, Any]]] = None,
    text_lookup: Optional[Dict[str, str]] = None,
    include_models: Optional[Sequence[str]] = None,
    gold_df: Optional[pd.DataFrame] = None,
    existing_embedding_cache: Optional[Dict[str, np.ndarray]] = None,
    compute_embeddings: bool = True,
    embed_model: str = "intfloat/multilingual-e5-large",
    embed_device: Optional[str] = None,
    embed_prefix: str = "passage: ",
    id_col: str = "file",
    model_col: str = "model",
    raw_output_col: str = "raw_output",
    doc_id_key: str = "file",
    doc_text_key: str = "text",
) -> LOOResources:
    """Create :class:`LOOResources` from baseline prediction logs.

    Parameters
    ----------
    predictions
        Wide table of zero-shot/ICL predictions with columns ``file``, ``model``
        and ``raw_output``.
    docs / text_lookup
        Either the raw document list (from :func:`load_documents`) or a mapping
        ``file -> text``.  One of them must be provided.
    include_models
        Optional filter to restrict the vote entropy computation to a subset of
        models (mirrors prior experimental setup).
    gold_df
        Optional gold labels for substring matching in selection utilities.
    existing_embedding_cache
        If supplied, used as-is.  Otherwise embeddings are computed once when
        ``compute_embeddings`` is True.
    """

    if text_lookup is None:
        if docs is None:
            raise ValueError("Either docs or text_lookup must be provided.")
        text_lookup = {str(d[doc_id_key]): d[doc_text_key] for d in docs}

    required_cols = {id_col, model_col, raw_output_col}
    missing = required_cols - set(predictions.columns)
    if missing:
        raise KeyError(f"Predictions DataFrame missing columns: {sorted(missing)}")

    work = predictions.copy()
    if include_models is not None:
        include = set(include_models)
        work = work[work[model_col].isin(include)]
    work[id_col] = work[id_col].astype(str)
    work[model_col] = work[model_col].astype(str)
    work["pred"] = work[raw_output_col].map(infer_label)

    vote_table = compute_vote_entropy(
        work,
        id_col=id_col,
        model_col=model_col,
        label_col="pred",
        normalize_entropy=True,
    )
    vote_table[id_col] = vote_table[id_col].astype(str)

    # Prediction vectors are used by granule-aware selectors.
    model_order = sorted(work[model_col].dropna().astype(str).unique().tolist())
    if model_order:
        vector_df = (
            work.pivot_table(index=id_col, columns=model_col, values="pred", aggfunc="first")
            .reindex(columns=model_order)
            .reset_index()
        )
        vector_df[id_col] = vector_df[id_col].astype(str)
        vector_df[model_order] = vector_df[model_order].fillna("__MISSING__").astype(str)
        vector_df["prediction_vector"] = vector_df[model_order].agg(" | ".join, axis=1)
        vote_table = vote_table.merge(
            vector_df[[id_col, "prediction_vector"]],
            on=id_col,
            how="left",
        )

    embedding_cache = existing_embedding_cache
    if compute_embeddings and embedding_cache is None:
        identifiers = vote_table[id_col].tolist()
        missing_text = [fid for fid in identifiers if fid not in text_lookup]
        if missing_text:
            raise KeyError(
                "Texts for the following identifiers are missing: "
                + ", ".join(sorted(missing_text)[:5])
                + ("..." if len(missing_text) > 5 else "")
            )
        embedder = build_e5_embedder(embed_model, device=embed_device, prefix=embed_prefix)
        vectors = embedder([text_lookup[fid] for fid in identifiers])
        vectors = np.asarray(vectors)
        embedding_cache = {fid: vectors[i] for i, fid in enumerate(identifiers)}

    return LOOResources(
        vote_table=vote_table.reset_index(drop=True),
        text_lookup=text_lookup,
        embedding_cache=embedding_cache,
        gold_df=gold_df,
        id_col=id_col,
    )


def run_leave_one_out_experiments(
    *,
    docs_dir: str,
    base_predictions_path: str,
    models: Sequence[str],
    output_dir: str,
    selection: str,
    selection_kwargs: Dict[str, Any],
    include_models: Optional[Sequence[str]] = None,
    gold_labels_path: Optional[str] = None,
    embed_model: str = "intfloat/multilingual-e5-large",
    embed_device: Optional[str] = None,
    embed_prefix: str = "passage: ",
    query_template: str = DEFAULT_QUERY_TEMPLATE,
    cache_prompts: bool = True,
    batch_precompute: bool = True,
) -> pd.DataFrame:
    """Run inference for a list of models using LOO exemplar prompts."""

    docs = load_documents(docs_dir)
    if not docs:
        raise RuntimeError(f"No documents found in {docs_dir}")

    base_predictions = pd.read_pickle(base_predictions_path)
    gold_df = None
    if gold_labels_path is not None:
        gold_df = pd.read_pickle(gold_labels_path)

    resources = prepare_loo_resources(
        base_predictions,
        docs=docs,
        include_models=include_models,
        gold_df=gold_df,
        embed_model=embed_model,
        embed_device=embed_device,
        embed_prefix=embed_prefix,
    )

    builder = LeaveOneOutPromptBuilder(
        resources,
        selection=selection,
        selection_kwargs=selection_kwargs,
        query_template=query_template,
        cache_prompts=cache_prompts,
    )

    if batch_precompute:
        builder.precompute_all()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_results: List[pd.DataFrame] = []
    for model_name in models:
        df = run_zero_shot_classification(
            docs,
            model_name,
            results_dir=str(output_path),
            PROMPT_TEMPLATE=builder,
        )
        all_results.append(df)

    if not all_results:
        return pd.DataFrame(columns=["file", "model", "raw_output"])

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_pickle(output_path / "all_models.pkl")
    return combined


def run_loo_and_evaluate(
    *,
    docs_dir: str,
    base_predictions_path: str,
    models: Sequence[str],
    output_dir: str,
    selection: str,
    selection_kwargs: Dict[str, Any],
    method_name: str,
    include_models: Optional[Sequence[str]] = None,
    gold_labels_path: str,
    embed_model: str = "intfloat/multilingual-e5-large",
    embed_device: Optional[str] = None,
    embed_prefix: str = "passage: ",
    query_template: str = DEFAULT_QUERY_TEMPLATE,
    cache_prompts: bool = True,
    batch_precompute: bool = True,
    metric: str = "f1",
    average: str = "macro",
    labels: Sequence[str] = ("unrelated", "late", "early"),
    B: int = 10000,
    stratified: bool = True,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Convenience wrapper that runs LOO inference and significance tests."""

    combined = run_leave_one_out_experiments(
        docs_dir=docs_dir,
        base_predictions_path=base_predictions_path,
        models=models,
        output_dir=output_dir,
        selection=selection,
        selection_kwargs=selection_kwargs,
        include_models=include_models,
        gold_labels_path=gold_labels_path,
        embed_model=embed_model,
        embed_device=embed_device,
        embed_prefix=embed_prefix,
        query_template=query_template,
        cache_prompts=cache_prompts,
        batch_precompute=batch_precompute,
    )

    gold_df = pd.read_pickle(gold_labels_path)
    summary, merged = evaluate_by_model(
        combined,
        gold_df,
        group_col="model",
        pred_col="raw_output",
        on="file",
        labels=labels,
    )

    merged_dfs = {method_name: merged}
    perf, pvals = evaluate_with_significance(
        merged_dfs,
        metric=metric,
        average=average,
        labels=labels,
        B=B,
        stratified=stratified,
        random_state=random_state,
        label_col="label",
        pred_col="pred",
        model_col="model",
        file_col="file_pred",
        system_id_cols=[],
        duplicate_file_strategy="vote",
    )

    return {
        "combined_predictions": combined,
        "per_model_summary": summary,
        "performance_table": perf,
        "p_values": pvals,
        "merged_predictions": merged,
    }


class LeaveOneOutPromptBuilder:
    """Callable that returns a leakage-free prompt for a given document."""

    def __init__(
        self,
        resources: LOOResources,
        *,
        selection: str = "ping",
        selection_kwargs: Optional[Dict[str, Any]] = None,
        prompt_kwargs: Optional[Dict[str, Any]] = None,
        query_template: str = DEFAULT_QUERY_TEMPLATE,
        text_key: str = "text",
        cache_prompts: bool = True,
        exclude_target_from_pool: bool = True,
    ) -> None:
        self.resources = resources
        self.selection = selection.lower()
        if self.selection in PING_SELECTIONS:
            self.selection_family = "ping"
        elif self.selection in RANDOM_SELECTIONS:
            self.selection_family = "random"
        else:
            self.selection_family = self.selection
        self.selection_kwargs = selection_kwargs.copy() if selection_kwargs else {}
        self.prompt_kwargs = prompt_kwargs.copy() if prompt_kwargs else {}
        self.query_template = query_template
        self.text_key = text_key
        self.cache_prompts = cache_prompts
        self.exclude_target_from_pool = bool(exclude_target_from_pool)

        table = resources.vote_table.copy()
        self.id_col = resources.id_col
        self.pred_col = resources.pred_col
        self.entropy_col = resources.entropy_col

        if self.pred_col not in table.columns:
            raise KeyError(
                f"Prediction column '{self.pred_col}' not found in vote_table."
            )
        if self.entropy_col not in table.columns:
            raise KeyError(
                f"Entropy column '{self.entropy_col}' not found in vote_table."
            )

        table[self.id_col] = table[self.id_col].astype(str)
        self._table_by_id = table.set_index(self.id_col, drop=False)
        self._prompt_cache: Dict[str, str] = {}
        self._selection_stats_by_target: Dict[str, Dict[str, Any]] = {}
        self._rationale_lookup: Dict[str, str] = {}

        # Pre-build a deterministic text loader to avoid disk IO in selectors
        self._text_lookup = resources.text_lookup
        if (
            resources.gold_df is not None
            and self.id_col in resources.gold_df.columns
            and "rationale" in resources.gold_df.columns
        ):
            rationale_df = resources.gold_df[[self.id_col, "rationale"]].copy()
            rationale_df[self.id_col] = rationale_df[self.id_col].astype(str)
            rationale_df["rationale"] = rationale_df["rationale"].fillna("").astype(str).str.strip()
            rationale_df = rationale_df[rationale_df["rationale"] != ""]
            rationale_df = rationale_df.drop_duplicates(subset=[self.id_col], keep="first")
            self._rationale_lookup = dict(zip(rationale_df[self.id_col], rationale_df["rationale"]))

    # ------------------------------------------------------------------
    def __call__(self, doc: Dict[str, Any]) -> str:
        return self.build(doc)

    def build(self, doc: Dict[str, Any]) -> str:
        doc_id = self.resources.ensure_id(doc)
        if self.cache_prompts and doc_id in self._prompt_cache:
            return self._prompt_cache[doc_id]

        doc_text = self.resources.ensure_text(doc_id, doc, text_key=self.text_key)

        if self.exclude_target_from_pool and doc_id in self._table_by_id.index:
            pool = self._table_by_id.drop(doc_id)
        else:
            pool = self._table_by_id

        if pool.empty:
            prompt = self.query_template.format(report_text=doc_text)
            if self.cache_prompts:
                self._prompt_cache[doc_id] = prompt
            return prompt

        pool = pool.reset_index(drop=True)
        base = self._build_prompt_body(pool, target_doc_id=doc_id)
        query_block = self.query_template.format(report_text=doc_text)
        if not base.strip():
            prompt = query_block
        else:
            prompt = base.rstrip() + "\n\n" + query_block
        if self.cache_prompts:
            self._prompt_cache[doc_id] = prompt
        return prompt

    def precompute_all(self, ids: Optional[Iterable[str]] = None) -> None:
        target_ids = ids or self._table_by_id.index.tolist()
        for fid in target_ids:
            if self.cache_prompts and fid in self._prompt_cache:
                continue
            doc = {self.id_col: fid, self.text_key: self._text_lookup[fid]}
            self.build(doc)

    # ------------------------------------------------------------------
    def _load_text(self, fid: str) -> str:
        try:
            return self._text_lookup[fid]
        except KeyError as exc:
            raise KeyError(f"Text missing for '{fid}'") from exc

    def _ensure_rationales_for_ids(self, selection_name: str, selected_ids: Sequence[str]) -> None:
        if not self._rationale_lookup:
            raise ValueError(
                f"selection='{selection_name}' requires gold_df with a non-empty 'rationale' column."
            )
        missing_rationales = [
            str(fid)
            for fid in selected_ids
            if not self._rationale_lookup.get(str(fid))
        ]
        if missing_rationales:
            preview = ", ".join(sorted(missing_rationales)[:5])
            if len(missing_rationales) > 5:
                preview += ", ..."
            raise ValueError(
                f"selection='{selection_name}' is missing rationales for exemplar files: "
                + preview
            )

    def _build_prompt_body(self, pool: pd.DataFrame, *, target_doc_id: Optional[str] = None) -> str:
        if self.selection_family == "ping":
            kwargs = self.selection_kwargs.copy()
            if "k" not in kwargs:
                raise ValueError("select_icls_pingpong requires 'k' in selection_kwargs")
            stats_payload: Dict[str, Any] = {}
            picks = select_icls_pingpong(
                pool,
                k=kwargs.pop("k"),
                id_col=self.id_col,
                pred_col=self.pred_col,
                entropy_col=self.entropy_col,
                text_loader=self._load_text,
                gold_df=self.resources.gold_df,
                embedding_cache=self.resources.embedding_cache,
                stats_out=stats_payload,
                **kwargs,
            )
            if target_doc_id is not None and stats_payload:
                self._selection_stats_by_target[target_doc_id] = stats_payload
            ping_prompt_allowed = {
                "instruction",
                "input_tag",
                "output_tag",
                "truncate_chars",
                "drop_if_no_gold",
                "add_query_placeholder",
                "label_map",
                "include_rationales",
                "rationale_tag",
            }
            prompt_kwargs = {
                key: value
                for key, value in self.prompt_kwargs.items()
                if key in ping_prompt_allowed
            }
            if self.selection in PING_RATIONALE_SELECTIONS:
                self._ensure_rationales_for_ids(
                    self.selection,
                    [str(item[0]) for item in picks],
                )
                prompt_kwargs["include_rationales"] = True
            return make_icl_prompt_from_pingpong(
                picks,
                add_query_placeholder=False,
                rationale_lookup=self._rationale_lookup,
                **prompt_kwargs,
            )

        if self.selection == "cbed":
            kwargs = self.selection_kwargs.copy()
            if "k_per_label" not in kwargs:
                raise ValueError(
                    "cbed_select_by_label requires 'k_per_label' in selection_kwargs"
                )
            results = cbed_select_by_label(
                pool,
                id_col=self.id_col,
                top_label_col=self.pred_col,
                entropy_col=self.entropy_col,
                text_loader=self._load_text,
                gold_df=self.resources.gold_df,
                embedding_cache=self.resources.embedding_cache,
                **kwargs,
            )
            return make_icl_prompt_from_results(
                results,
                add_query_placeholder=False,
                **self.prompt_kwargs,
            )

        if self.selection == "topk":
            kwargs = self.selection_kwargs.copy()
            if "k" not in kwargs:
                raise ValueError("topk_by_top_label requires 'k' in selection_kwargs")
            table = topk_by_top_label(
                pool,
                k=kwargs.pop("k"),
                id_col=self.id_col,
                top_label_col=self.pred_col,
                entropy_col=self.entropy_col,
                text_loader=self._load_text,
                gold_df=self.resources.gold_df,
                **kwargs,
            )
            return make_icl_prompt_from_results(
                table,
                add_query_placeholder=False,
                **self.prompt_kwargs,
            )

        if self.selection_family == "random":
            kwargs = self.selection_kwargs.copy()
            if "k" not in kwargs:
                raise ValueError("randomk_by_gold_label requires 'k' in selection_kwargs")
            bag = randomk_by_gold_label(
                pool,
                k=kwargs.pop("k"),
                id_col=self.id_col,
                entropy_col=self.entropy_col,
                text_loader=self._load_text,
                gold_df=self.resources.gold_df,
                **kwargs,
            )
            prompt_kwargs = self.prompt_kwargs.copy()
            if self.selection == "random_ra":
                self._ensure_rationales_for_ids(
                    "random_ra",
                    [str(item[0]) for items in bag.values() for item in items],
                )
                prompt_kwargs["include_rationales"] = True
            return make_icl_prompt_from_results(
                bag,
                add_query_placeholder=False,
                rationale_lookup=self._rationale_lookup,
                **prompt_kwargs,
            )

        raise ValueError(
            "Unsupported selection. Choose from "
            "{'ping','ping_ra','cross_ping','cross_ping_ra','cbed','topk','random','random_ra'}."
        )

    def ping_selection_stats_frames(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Return per-target summary and per-exemplar event tables for ping selection."""
        if self.selection_family != "ping" or not self._selection_stats_by_target:
            return pd.DataFrame(), pd.DataFrame()

        summary_rows: List[Dict[str, Any]] = []
        event_rows: List[Dict[str, Any]] = []
        for target_file, payload in self._selection_stats_by_target.items():
            summary = dict(payload.get("summary", {}))
            summary_rows.append({"target_file": target_file, **summary})

            events = payload.get("events", [])
            for event in events:
                if not isinstance(event, dict):
                    continue
                event_rows.append({"target_file": target_file, **event})

        summary_df = pd.DataFrame(summary_rows)
        events_df = pd.DataFrame(event_rows)
        return summary_df, events_df


# Convenience -----------------------------------------------------------------

@lru_cache(maxsize=None)
def cached_embedding_cache(
    ids_tuple: tuple,
    texts_tuple: tuple,
    *,
    model_name: str = "intfloat/multilingual-e5-large",
    device: Optional[str] = None,
    prefix: str = "passage: ",
) -> Dict[str, np.ndarray]:
    """LRU-cached helper to build an embedding cache (primarily for tests)."""

    embedder = build_e5_embedder(model_name, device=device, prefix=prefix)
    vectors = embedder(list(texts_tuple))
    vectors = np.asarray(vectors)
    return {fid: vectors[i] for i, fid in enumerate(ids_tuple)}
