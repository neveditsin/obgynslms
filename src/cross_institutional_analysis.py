"""
Cross-Institutional ICL Transfer Analysis
==========================================
Produces all statistics needed for Results Section 2.3:

OUTPUT 1 — transfer_table_1:
    Macro F1 [95% bootstrap CI] for each model + committee, both cohorts,
    both prompt variants, under cross-institutional label-only and
    rationale-augmented ICL (WithUpdate selection held constant).

OUTPUT 2 — transfer_table_2:
    Per-class (early, late, unrelated) precision / recall / F1 for each
    model under transfer conditions, at the best prompt variant (long).

OUTPUT 3 — transfer_table_3:
    Paired bootstrap comparisons: local vs. transfer for each model,
    separately for label-only and rationale-augmented.
    Δ = local − transfer; positive means local was better.

OUTPUT 4 — transfer_table_4:
    Summary: mean absolute Δ across models per cohort × prompt regime,
    count of significant degradations, count of significant improvements.

All tables are saved to analysis/cross_institutional_tables/ as CSVs.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from IPython.display import display

# ── Imports from existing pipeline ────────────────────────────────────
from src.consolidated_loo_eval import (
    PROJECT_ROOT,
    MACRO_LABELS,
    EXPECTED_SEEDS,
    COMMITTEE_MODEL_NAME,
    discover_variant_specs,
    load_icl_predictions,
    merge_with_gold,
    _collapse_model_predictions,
    _build_seed_level_committee,
    _collapse_committee_predictions,
    _filter_primary_condition,
    _encode_predictions,
    bootstrap_macro_f1_ci,
    paired_bootstrap_comparison,
    _stable_seed,
    load_gold_labels,
    ConditionSpec,
    PROMPT_REGIME_LABEL_ONLY,
    PROMPT_REGIME_RATIONALE,
)
from sklearn.metrics import precision_recall_fscore_support

# ── Config ────────────────────────────────────────────────────────────
PRIMARY_M = 4
BOOTSTRAP_REPS = 5000
ALPHA = 0.05
RANDOM_STATE = 42          # different seed from main tables to avoid overlap
DISPLAY_FULL = True
OUTPUT_DIR = Path("analysis/cross_institutional_tables")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"PRIMARY_M     = {PRIMARY_M}")
print(f"BOOTSTRAP_REPS = {BOOTSTRAP_REPS}")
print(f"ALPHA          = {ALPHA}")
print(f"RANDOM_STATE   = {RANDOM_STATE}")

# ── Cross-institutional condition specs ───────────────────────────────
CROSS_CONDITIONS = [
    ConditionSpec(
        key="cross_label_only",
        source_key="cross_ping",
        regime="Transfer",
        prompt_regime=PROMPT_REGIME_LABEL_ONLY,
        expect_five_seeds=True,
    ),
    ConditionSpec(
        key="cross_rationale",
        source_key="cross_ping_ra",
        regime="Transfer",
        prompt_regime=PROMPT_REGIME_RATIONALE,
        expect_five_seeds=True,
    ),
]

# Local conditions to compare against
LOCAL_CONDITIONS = [
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
]

# Pairing: (local_condition, cross_condition, comparison_label)
COMPARISON_PAIRS = [
    (LOCAL_CONDITIONS[0], CROSS_CONDITIONS[0], "local_label_only vs transfer_label_only"),
    (LOCAL_CONDITIONS[1], CROSS_CONDITIONS[1], "local_rationale vs transfer_rationale"),
]


def display_table(df):
    if DISPLAY_FULL:
        with pd.option_context(
            "display.max_rows", None,
            "display.max_columns", None,
            "display.max_colwidth", None,
            "display.width", None,
        ):
            display(df)
    else:
        display(df)


# ── Load variant specs ────────────────────────────────────────────────
specs = discover_variant_specs()
print(f"\nLoaded {len(specs)} variant specs:")
for s in specs:
    print(f"  {s.cohort} / {s.prompt_variant} → {s.output_root}")


# ══════════════════════════════════════════════════════════════════════
# STEP 1: Coverage check — do cross_ping / cross_ping_ra folders exist
#         and have complete 5-seed coverage at PRIMARY_M?
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("COVERAGE CHECK")
print("=" * 70)

coverage_rows = []
for spec in specs:
    gold_df = load_gold_labels(spec.gold_path)
    for cond in CROSS_CONDITIONS:
        raw = load_icl_predictions(spec, cond.source_key)
        if raw.empty:
            print(f"  ⚠ EMPTY: {spec.cohort}/{spec.prompt_variant}/{cond.source_key}")
            coverage_rows.append({
                "cohort": spec.cohort,
                "prompt_variant": spec.prompt_variant,
                "source_key": cond.source_key,
                "n_raw_rows": 0,
                "available_seeds": [],
                "available_m": [],
                "is_complete": False,
            })
            continue

        merged = merge_with_gold(raw, gold_df)
        primary = merged[merged["m"].astype(int) == PRIMARY_M] if not merged.empty else merged
        seeds_found = sorted(primary["seed"].dropna().astype(int).unique().tolist()) if not primary.empty else []
        m_found = sorted(raw["m"].dropna().astype(int).unique().tolist()) if not raw.empty else []
        is_complete = seeds_found == sorted(EXPECTED_SEEDS)

        coverage_rows.append({
            "cohort": spec.cohort,
            "prompt_variant": spec.prompt_variant,
            "source_key": cond.source_key,
            "n_raw_rows": len(raw),
            "available_seeds": seeds_found,
            "available_m": m_found,
            "is_complete": is_complete,
        })
        status = "✓" if is_complete else "⚠ INCOMPLETE"
        print(f"  {status}: {spec.cohort}/{spec.prompt_variant}/{cond.source_key}  "
              f"seeds={seeds_found}  m_values={m_found}  rows={len(raw)}")

coverage_df = pd.DataFrame(coverage_rows)
display_table(coverage_df)


# ══════════════════════════════════════════════════════════════════════
# STEP 2: Build transfer_table_1 — macro F1 with bootstrap CIs
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TRANSFER TABLE 1: Macro F1 with 95% Bootstrap CIs")
print("=" * 70)

table_1_rows = []
# Store collapsed predictions for later use in comparisons
prediction_store = {}

for spec in specs:
    gold_df = load_gold_labels(spec.gold_path)

    for cond in CROSS_CONDITIONS + LOCAL_CONDITIONS:
        raw = load_icl_predictions(spec, cond.source_key)
        if raw.empty:
            continue

        merged = merge_with_gold(raw, gold_df)
        primary, is_complete = _filter_primary_condition(
            merged, cond, primary_m=PRIMARY_M, expected_seeds=EXPECTED_SEEDS
        )
        if primary.empty:
            continue

        # Collapse per-model predictions (majority vote over seeds)
        model_preds = _collapse_model_predictions(primary)

        # Build committee
        seed_committee = _build_seed_level_committee(primary)
        collapsed_committee = _collapse_committee_predictions(seed_committee)

        for model_name in list(spec.models) + [COMMITTEE_MODEL_NAME]:
            if model_name == COMMITTEE_MODEL_NAME:
                model_frame = collapsed_committee
            else:
                model_frame = model_preds[model_preds["model"] == model_name].copy()

            if model_frame.empty:
                continue

            # Store for paired comparisons
            store_key = (spec.cohort, spec.prompt_variant, cond.key, model_name)
            prediction_store[store_key] = model_frame

            observed, ci_low, ci_high, n_docs = bootstrap_macro_f1_ci(
                model_frame,
                B=BOOTSTRAP_REPS,
                alpha=ALPHA,
                random_state=_stable_seed(
                    "cross", spec.cohort, spec.prompt_variant, cond.key, model_name,
                    base=RANDOM_STATE,
                ),
            )
            table_1_rows.append({
                "cohort": spec.cohort,
                "prompt_variant": spec.prompt_variant,
                "model": model_name,
                "regime": cond.regime,
                "prompt_regime": cond.prompt_regime,
                "macro_f1": observed,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n_docs": n_docs,
            })

transfer_table_1 = pd.DataFrame(table_1_rows)
print(f"\nGenerated {len(transfer_table_1)} rows")
display_table(transfer_table_1)
transfer_table_1.to_csv(OUTPUT_DIR / "transfer_table_1.csv", index=False)


# ══════════════════════════════════════════════════════════════════════
# STEP 3: Build transfer_table_2 — per-class P/R/F1 for transfer
#         conditions (long prompt variant only, both cohorts)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TRANSFER TABLE 2: Per-Class Metrics (long prompt)")
print("=" * 70)

table_2_rows = []
for spec in specs:
    if spec.prompt_variant != "long":
        continue
    gold_df = load_gold_labels(spec.gold_path)

    for cond in CROSS_CONDITIONS + LOCAL_CONDITIONS:
        raw = load_icl_predictions(spec, cond.source_key)
        if raw.empty:
            continue
        merged = merge_with_gold(raw, gold_df)
        primary, _ = _filter_primary_condition(
            merged, cond, primary_m=PRIMARY_M, expected_seeds=EXPECTED_SEEDS
        )
        if primary.empty:
            continue

        model_preds = _collapse_model_predictions(primary)
        seed_committee = _build_seed_level_committee(primary)
        collapsed_committee = _collapse_committee_predictions(seed_committee)

        for model_name in list(spec.models) + [COMMITTEE_MODEL_NAME]:
            if model_name == COMMITTEE_MODEL_NAME:
                mf = collapsed_committee
            else:
                mf = model_preds[model_preds["model"] == model_name].copy()
            if mf.empty:
                continue

            valid = mf.dropna(subset=["label", "pred"])
            if valid.empty:
                continue

            prec, rec, f1, sup = precision_recall_fscore_support(
                valid["label"], valid["pred"],
                labels=MACRO_LABELS,
                average=None,
                zero_division=0,
            )
            for i, cls in enumerate(MACRO_LABELS):
                table_2_rows.append({
                    "cohort": spec.cohort,
                    "model": model_name,
                    "regime": cond.regime,
                    "prompt_regime": cond.prompt_regime,
                    "class": cls,
                    "precision": prec[i],
                    "recall": rec[i],
                    "f1": f1[i],
                    "support": int(sup[i]),
                })

transfer_table_2 = pd.DataFrame(table_2_rows)
print(f"\nGenerated {len(transfer_table_2)} rows")
display_table(transfer_table_2)
transfer_table_2.to_csv(OUTPUT_DIR / "transfer_table_2.csv", index=False)


# ══════════════════════════════════════════════════════════════════════
# STEP 4: Build transfer_table_3 — paired bootstrap: local vs transfer
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TRANSFER TABLE 3: Paired Bootstrap — Local vs Transfer")
print("=" * 70)

table_3_rows = []
for spec in specs:
    for local_cond, cross_cond, comparison_label in COMPARISON_PAIRS:
        for model_name in list(spec.models) + [COMMITTEE_MODEL_NAME]:
            local_frame = prediction_store.get(
                (spec.cohort, spec.prompt_variant, local_cond.key, model_name)
            )
            cross_frame = prediction_store.get(
                (spec.cohort, spec.prompt_variant, cross_cond.key, model_name)
            )

            stats = paired_bootstrap_comparison(
                local_frame if isinstance(local_frame, pd.DataFrame) else pd.DataFrame(),
                cross_frame if isinstance(cross_frame, pd.DataFrame) else pd.DataFrame(),
                B=BOOTSTRAP_REPS,
                alpha=ALPHA,
                random_state=_stable_seed(
                    "cross_compare", spec.cohort, spec.prompt_variant,
                    comparison_label, model_name,
                    base=RANDOM_STATE,
                ),
            )
            table_3_rows.append({
                "cohort": spec.cohort,
                "prompt_variant": spec.prompt_variant,
                "model": model_name,
                "comparison": comparison_label,
                "delta_f1": stats["delta_f1"],
                "bootstrap_ci_low": stats["bootstrap_ci_low"],
                "bootstrap_ci_high": stats["bootstrap_ci_high"],
                "bootstrap_p": stats["bootstrap_p"],
                "significant": bool(stats["bootstrap_p"] < ALPHA)
                if pd.notna(stats["bootstrap_p"])
                else False,
            })

transfer_table_3 = pd.DataFrame(table_3_rows)
print(f"\nGenerated {len(transfer_table_3)} rows")
display_table(transfer_table_3)
transfer_table_3.to_csv(OUTPUT_DIR / "transfer_table_3.csv", index=False)


# ══════════════════════════════════════════════════════════════════════
# STEP 5: Build transfer_table_4 — summary statistics
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TRANSFER TABLE 4: Summary of Transfer Effects")
print("=" * 70)

# Exclude committee from summary stats (report separately)
t3_models_only = transfer_table_3[
    transfer_table_3["model"] != COMMITTEE_MODEL_NAME
].copy()

summary_rows = []
for (cohort, pv, comparison), grp in t3_models_only.groupby(
    ["cohort", "prompt_variant", "comparison"]
):
    n_total = len(grp)
    n_sig = grp["significant"].sum()
    # Positive delta = local better; negative = transfer better
    n_local_better_sig = ((grp["delta_f1"] > 0) & grp["significant"]).sum()
    n_transfer_better_sig = ((grp["delta_f1"] < 0) & grp["significant"]).sum()
    n_no_sig = n_total - n_sig

    summary_rows.append({
        "cohort": cohort,
        "prompt_variant": pv,
        "comparison": comparison,
        "n_models": n_total,
        "mean_delta_f1": grp["delta_f1"].mean(),
        "median_delta_f1": grp["delta_f1"].median(),
        "mean_abs_delta_f1": grp["delta_f1"].abs().mean(),
        "max_abs_delta_f1": grp["delta_f1"].abs().max(),
        "n_no_significant_diff": int(n_no_sig),
        "n_local_significantly_better": int(n_local_better_sig),
        "n_transfer_significantly_better": int(n_transfer_better_sig),
    })

transfer_table_4 = pd.DataFrame(summary_rows)
display_table(transfer_table_4)
transfer_table_4.to_csv(OUTPUT_DIR / "transfer_table_4.csv", index=False)


# ══════════════════════════════════════════════════════════════════════
# STEP 6: Formatted summary export
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("FORMATTED SUMMARY (long prompt, rationale-augmented)")
print("=" * 70)

for cohort_name in ["MIMIC", "Indian"]:
    print(f"\n--- {cohort_name} cohort ---")

    # Transfer direction label
    if cohort_name == "MIMIC":
        transfer_dir = "Indian → MIMIC"
    else:
        transfer_dir = "MIMIC → Indian"

    # Local vs transfer for rationale condition
    sub = transfer_table_3[
        (transfer_table_3["cohort"] == cohort_name)
        & (transfer_table_3["prompt_variant"] == "long")
        & (transfer_table_3["comparison"] == "local_rationale vs transfer_rationale")
    ].copy()

    if sub.empty:
        print(f"  No data for {cohort_name} long rationale comparison")
        continue

    for _, row in sub.iterrows():
        model_short = row["model"].split("/")[-1] if "/" in str(row["model"]) else row["model"]
        sig_marker = "*" if row["significant"] else ""
        print(f"  {model_short:40s}  Δ = {row['delta_f1']:+.3f}  "
              f"[{row['bootstrap_ci_low']:+.3f}, {row['bootstrap_ci_high']:+.3f}]  "
              f"p={row['bootstrap_p']:.4f}{sig_marker}")

    # Also show the transfer F1 values
    print(f"\n  Transfer F1 ({transfer_dir}, rationale-augmented):")
    sub_f1 = transfer_table_1[
        (transfer_table_1["cohort"] == cohort_name)
        & (transfer_table_1["prompt_variant"] == "long")
        & (transfer_table_1["regime"] == "Transfer")
        & (transfer_table_1["prompt_regime"] == PROMPT_REGIME_RATIONALE)
    ].copy()
    for _, row in sub_f1.iterrows():
        model_short = row["model"].split("/")[-1] if "/" in str(row["model"]) else row["model"]
        print(f"    {model_short:40s}  F1 = {row['macro_f1']:.3f}  "
              f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}]")


print("\n" + "=" * 70)
print("PAPER-READY SUMMARY (long prompt, label-only)")
print("=" * 70)

for cohort_name in ["MIMIC", "Indian"]:
    print(f"\n--- {cohort_name} cohort ---")
    if cohort_name == "MIMIC":
        transfer_dir = "Indian → MIMIC"
    else:
        transfer_dir = "MIMIC → Indian"

    sub = transfer_table_3[
        (transfer_table_3["cohort"] == cohort_name)
        & (transfer_table_3["prompt_variant"] == "long")
        & (transfer_table_3["comparison"] == "local_label_only vs transfer_label_only")
    ].copy()

    if sub.empty:
        print(f"  No data for {cohort_name} long label-only comparison")
        continue

    for _, row in sub.iterrows():
        model_short = row["model"].split("/")[-1] if "/" in str(row["model"]) else row["model"]
        sig_marker = "*" if row["significant"] else ""
        print(f"  {model_short:40s}  Δ = {row['delta_f1']:+.3f}  "
              f"[{row['bootstrap_ci_low']:+.3f}, {row['bootstrap_ci_high']:+.3f}]  "
              f"p={row['bootstrap_p']:.4f}{sig_marker}")

    print(f"\n  Transfer F1 ({transfer_dir}, label-only):")
    sub_f1 = transfer_table_1[
        (transfer_table_1["cohort"] == cohort_name)
        & (transfer_table_1["prompt_variant"] == "long")
        & (transfer_table_1["regime"] == "Transfer")
        & (transfer_table_1["prompt_regime"] == PROMPT_REGIME_LABEL_ONLY)
    ].copy()
    for _, row in sub_f1.iterrows():
        model_short = row["model"].split("/")[-1] if "/" in str(row["model"]) else row["model"]
        print(f"    {model_short:40s}  F1 = {row['macro_f1']:.3f}  "
              f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}]")


# ══════════════════════════════════════════════════════════════════════
# STEP 7: Per-class degradation analysis
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PER-CLASS ANALYSIS: Which classes degrade most under transfer?")
print("=" * 70)

t2 = transfer_table_2.copy()
for cohort_name in ["MIMIC", "Indian"]:
    print(f"\n--- {cohort_name} cohort (long prompt, rationale-augmented) ---")

    local_sub = t2[
        (t2["cohort"] == cohort_name)
        & (t2["regime"] == "WithUpdate")
        & (t2["prompt_regime"] == PROMPT_REGIME_RATIONALE)
    ].copy()
    transfer_sub = t2[
        (t2["cohort"] == cohort_name)
        & (t2["regime"] == "Transfer")
        & (t2["prompt_regime"] == PROMPT_REGIME_RATIONALE)
    ].copy()

    if local_sub.empty or transfer_sub.empty:
        print("  Insufficient data")
        continue

    # Merge on model + class to compute per-class deltas
    merged = local_sub.merge(
        transfer_sub,
        on=["cohort", "model", "class"],
        suffixes=("_local", "_transfer"),
        how="inner",
    )
    if merged.empty:
        print("  No overlapping models")
        continue

    merged["delta_f1"] = merged["f1_local"] - merged["f1_transfer"]

    # Average across models per class
    class_summary = (
        merged[merged["model"] != COMMITTEE_MODEL_NAME]
        .groupby("class")["delta_f1"]
        .agg(mean_delta="mean", median_delta="median", min_delta="min", max_delta="max")
        .reset_index()
    )
    print("  Mean Δ F1 per class (local − transfer), averaged across models:")
    for _, row in class_summary.iterrows():
        print(f"    {row['class']:12s}  mean Δ = {row['mean_delta']:+.3f}  "
              f"range [{row['min_delta']:+.3f}, {row['max_delta']:+.3f}]")


print(f"\n\nAll tables saved to: {OUTPUT_DIR.resolve()}")
print("Files:")
for p in sorted(OUTPUT_DIR.glob("*.csv")):
    print(f"  {p.name}")
