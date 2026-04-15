#!/usr/bin/env python3
"""Evaluate regex_baseline.py on the configured corpus from config paths.

Usage:
    python evaluate_mimic_regex_baseline.py \
      --config configs/mimic_baseline_loo_config.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from regex_baseline import EARLY, LATE, UNRELATED, evaluate_corpus
from src.config_utils import resolve_dataset_paths as _resolve_dataset_paths_cfg
from src.data_loader import load_documents

SHORT_LABELS = ["early", "late", "unrelated"]
BASELINE_MODEL_NAME = "regex_baseline"


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_dataset_paths(
    paths_cfg: Dict[str, Any],
    *,
    base_dir: Optional[Path] = None,
) -> Tuple[Path, Optional[Path], Dict[str, Any]]:
    return _resolve_dataset_paths_cfg(paths_cfg, base_dir=base_dir)


def _load_gold_labels_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        frame = pd.read_pickle(path)
    elif suffix in {".tsv", ".tab"}:
        frame = pd.read_csv(path, sep="\t")
    else:
        frame = pd.read_csv(path)

    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"Gold labels did not load as a DataFrame: {path}")
    if "file" not in frame.columns or "label" not in frame.columns:
        raise KeyError(f"Gold labels must contain 'file' and 'label': {path}")
    return frame


def _normalize_gold_to_baseline(label: str) -> str:
    value = str(label).strip().lower()
    mapping = {
        EARLY: EARLY,
        "early": EARLY,
        "early pregnancy": EARLY,
        "early active pregnancy": EARLY,
        LATE: LATE,
        "late": LATE,
        "late pregnancy": LATE,
        "late active pregnancy": LATE,
        UNRELATED: UNRELATED,
        "unrelated": UNRELATED,
        "no": UNRELATED,
        "no active": UNRELATED,
        "no active pregnancy": UNRELATED,
        "no current pregnancy": UNRELATED,
        "unrelated or no current pregnancy": UNRELATED,
    }
    if value not in mapping:
        raise ValueError(
            f"Unsupported gold label '{label}'. Expected one of: {sorted(mapping)}"
        )
    return mapping[value]


def _normalize_to_short(label: str) -> str:
    value = str(label).strip().lower()
    if value in {EARLY, "early", "early pregnancy", "early active pregnancy"}:
        return "early"
    if value in {LATE, "late", "late pregnancy", "late active pregnancy"}:
        return "late"
    if value in {
        UNRELATED,
        "unrelated",
        "no",
        "no active",
        "no active pregnancy",
        "no current pregnancy",
        "unrelated or no current pregnancy",
    }:
        return "unrelated"
    return "UNK"


def _prepare_eval_frame(docs: List[Dict[str, str]], gold_df: pd.DataFrame) -> pd.DataFrame:
    docs_df = pd.DataFrame(docs)
    if docs_df.empty:
        raise RuntimeError("No documents found in configured data_dir.")

    if "file" not in docs_df.columns or "text" not in docs_df.columns:
        raise KeyError("Loaded docs must contain 'file' and 'text' columns.")

    if "file" not in gold_df.columns or "label" not in gold_df.columns:
        raise KeyError("Gold labels must contain 'file' and 'label' columns.")

    docs_df = docs_df.copy()
    docs_df["file"] = docs_df["file"].astype(str)
    docs_df["file_key"] = docs_df["file"].str.lower()
    docs_df = docs_df.drop_duplicates(subset=["file_key"], keep="first")

    gold = gold_df.copy()
    gold["file"] = gold["file"].astype(str)
    gold["gold_short"] = gold["label"].map(_normalize_to_short)
    gold["gold_baseline"] = gold["label"].map(_normalize_gold_to_baseline)

    doc_names_orig = docs_df["file"].tolist()
    doc_names_norm = docs_df["file_key"].tolist()

    def _best_doc_match(file_value: str) -> Optional[str]:
        key = str(file_value).strip().lower()
        if not key:
            return None
        best_doc = None
        best_len = -1
        for orig, norm in zip(doc_names_orig, doc_names_norm):
            if (key in norm or norm in key) and len(norm) > best_len:
                best_doc = orig
                best_len = len(norm)
        return best_doc

    gold["doc_file"] = gold["file"].map(_best_doc_match)
    merged = gold.merge(
        docs_df[["file", "text"]].rename(columns={"file": "doc_file"}),
        on="doc_file",
        how="left",
    )

    missing_text = merged["text"].isna()
    if missing_text.any():
        missing_files = merged.loc[missing_text, "file"].head(20).tolist()
        preview = ", ".join(missing_files)
        raise RuntimeError(
            "Missing source text for gold-labeled files (showing up to 20): "
            f"{preview}"
        )

    return merged.sort_values("file").reset_index(drop=True)


def _compute_short_metrics(eval_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y_true = eval_df["gold_short"].tolist()
    y_pred = eval_df["pred_short"].tolist()

    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=SHORT_LABELS,
        average="macro",
        zero_division=0,
    )

    per_p, per_r, per_f1, per_support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=SHORT_LABELS,
        average=None,
        zero_division=0,
    )

    per_class = pd.DataFrame(
        {
            "label": SHORT_LABELS,
            "precision": per_p,
            "recall": per_r,
            "f1": per_f1,
            "support": per_support,
        }
    )

    cm = confusion_matrix(y_true, y_pred, labels=SHORT_LABELS)
    confusion = pd.DataFrame(cm, index=SHORT_LABELS, columns=SHORT_LABELS)

    summary = pd.DataFrame(
        [
            {
                "model": BASELINE_MODEL_NAME,
                "precision": float(p_macro),
                "recall": float(r_macro),
                "f1": float(f1_macro),
                "n": int(len(eval_df)),
            }
        ]
    )

    return summary, per_class, confusion


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate regex baseline on configured corpus using config paths."
    )
    parser.add_argument(
        "--config",
        default="configs/mimic_baseline_loo_config.json",
        help="Path to JSON config file.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to <paths.output_root>/regex_baseline.",
    )
    parser.add_argument(
        "--output-prefix",
        default="regex_baseline",
        help="Filename prefix for generated artifacts.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed baseline evaluation output.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = _load_json(config_path)
    paths_cfg = cfg.get("paths", {})
    data_dir, gold_path, paths_cfg = _resolve_dataset_paths(
        paths_cfg,
        base_dir=config_path.parent.resolve(),
    )
    output_root = Path(paths_cfg.get("output_root", "results/mimic_streamlined_pipeline_small_models"))

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    if gold_path is None:
        raise ValueError("Config must provide gold labels for evaluation.")
    if not gold_path.exists():
        raise FileNotFoundError(f"Gold labels file not found: {gold_path}")

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = output_root / "regex_baseline"
    output_dir.mkdir(parents=True, exist_ok=True)

    docs = load_documents(str(data_dir))
    gold_df = _load_gold_labels_frame(gold_path)
    eval_frame = _prepare_eval_frame(docs, gold_df)

    texts = eval_frame["text"].tolist()
    gold_baseline = eval_frame["gold_baseline"].tolist()

    baseline_results = evaluate_corpus(texts, gold_baseline, verbose=args.verbose)
    predictions_baseline = baseline_results["predictions"]
    details = baseline_results["detailed_results"]

    eval_frame = eval_frame.assign(
        pred_baseline=predictions_baseline,
        pred_short=[_normalize_to_short(x) for x in predictions_baseline],
        confidence=[r.confidence for r in details],
        rule_fired=[r.rule_fired for r in details],
        ga_weeks=[r.ga_weeks for r in details],
        early_marker_count=[r.early_marker_count for r in details],
        late_marker_count=[r.late_marker_count for r in details],
        correct=lambda d: d["gold_short"] == d["pred_short"],
    )

    model_summary, per_class, confusion = _compute_short_metrics(eval_frame)
    accuracy = float((eval_frame["gold_short"] == eval_frame["pred_short"]).mean())

    error_by_rule = (
        eval_frame.loc[~eval_frame["correct"]]
        .groupby("rule_fired")
        .size()
        .reset_index(name="errors")
        .sort_values("errors", ascending=False)
    )
    error_by_pair = (
        eval_frame.loc[~eval_frame["correct"]]
        .groupby(["gold_short", "pred_short"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    prefix = args.output_prefix
    pred_cols = [
        "file",
        "doc_file",
        "gold_short",
        "gold_baseline",
        "pred_short",
        "pred_baseline",
        "correct",
        "confidence",
        "rule_fired",
        "ga_weeks",
        "early_marker_count",
        "late_marker_count",
        "text",
    ]
    predictions_df = eval_frame[pred_cols].copy()

    predictions_df.to_pickle(output_dir / f"{prefix}_predictions.pkl")
    predictions_df.to_csv(output_dir / f"{prefix}_predictions.csv", index=False)
    model_summary.to_csv(output_dir / f"{prefix}_model_summary.csv", index=False)
    per_class.to_csv(output_dir / f"{prefix}_per_class.csv", index=False)
    confusion.to_csv(output_dir / f"{prefix}_confusion_matrix.csv")
    error_by_rule.to_csv(output_dir / f"{prefix}_error_by_rule.csv", index=False)
    error_by_pair.to_csv(output_dir / f"{prefix}_error_by_label_pair.csv", index=False)

    report = {
        "model": BASELINE_MODEL_NAME,
        "n_docs": int(len(eval_frame)),
        "accuracy": accuracy,
        "macro_precision": float(model_summary.loc[0, "precision"]),
        "macro_recall": float(model_summary.loc[0, "recall"]),
        "macro_f1": float(model_summary.loc[0, "f1"]),
        "baseline_native_accuracy": float(baseline_results["accuracy"]),
        "baseline_native_macro_f1": float(baseline_results["macro_f1"]),
        "config": str(config_path),
        "data_dir": str(data_dir),
        "gold_labels": str(gold_path),
        "output_dir": str(output_dir),
    }
    with (output_dir / f"{prefix}_summary.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Regex baseline evaluation complete.")
    print(f"Documents evaluated: {report['n_docs']}")
    print(f"Accuracy: {report['accuracy']:.4f}")
    print(f"Macro-F1: {report['macro_f1']:.4f}")
    print(f"Wrote artifacts to: {output_dir}")


if __name__ == "__main__":
    main()
