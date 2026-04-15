# OBGYN SLMs Reproducibility Package

This repository is the reproducibility-oriented code package for reproducing the experiments in this repository. It is intentionally code-first:

- core implementation lives in `src/`
- experiment entry points are `run_mimic_baseline_and_loo.py` and `evaluate_mimic_regex_baseline.py`
- reproducibility template configs live in `configs/`
- generated numeric outputs are not stored in the repository

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

Optional notebook dependencies:

```bash
pip install -e ".[notebooks]"
```

After installation, the main CLIs are available as:

```bash
obgynslms-run --help
obgynslms-regex-baseline --help
```

You can also call the scripts directly with `python`.

## Data Layout

Paths in the reproducibility configs are resolved relative to the config file location. The shipped configs expect this directory layout:

```text
reproducibility_data/
  mimic/
    documents/
      <report_1>.txt
      <report_2>.txt
      ...
    gold_labels.csv
  india/
    documents/
      <report_1>.txt
      <report_2>.txt
      ...
    gold_labels.csv
```

### Document directory

- The loader reads `*.txt` files only.
- Each file is one report.
- Filenames must remain stable because prediction-to-gold matching uses the filename.
- The runtime document object has two fields: `file` and `text`.

### Gold labels file

Supported formats:

- `.csv`
- `.tsv`
- `.pkl` / `.pickle` containing a pandas DataFrame

Required columns:

- `file`: filename or filename-like identifier matching the report text file
- `label`: one of `early active pregnancy`, `late active pregnancy`, `no active pregnancy`

Conditionally required column:

- `rationale`: required for rationale-augmented selections `random_ra`, `ping_ra`, and `cross_ping_ra`

Example:

```csv
file,label,rationale
doc_001.txt,early active pregnancy,Gestational sac and yolk sac are present.
doc_002.txt,late active pregnancy,The report describes a second-trimester anatomy scan.
doc_003.txt,no active pregnancy,The report documents a postpartum or resolved state.
```

### Base prediction artifact for LOO / cross-dataset transfer

Cross-dataset selectors require the opposite cohort's zero-shot predictions. The expected artifact is:

- `zero_shot/all_models.pkl`

That pickle should contain at least:

- `file`
- `model`
- `raw_output`

The zero-shot stage in this repo generates that artifact automatically.

## Config Guide

Reproducibility templates:

- `configs/reproducibility_minimal_mimic_long.json`
- `configs/reproducibility_minimal_indic_long.json`

Important config sections:

- `paths.dataset`: active cohort key, either `mimic` or `our`
- `paths.data_dir`: directory of report `.txt` files
- `paths.gold_labels`: gold labels file for the active cohort
- `paths.output_root`: where results for the active run are written
- `paths.dataset_paths.<dataset>.base_predictions_path`: opposite-cohort zero-shot predictions used by `cross_ping` and `cross_ping_ra`
- `models`: Hugging Face model identifiers to evaluate
- `zero_shot`: prompt settings and zero-shot output location
- `loo.run_selections`: any subset of `ping`, `ping_ra`, `cross_ping`, `cross_ping_ra`, `random`, `random_ra`
- `loo.k_values`: exemplar counts
- `loo.seeds`: random seeds for stochastic selectors

## Reproduce The Experiments

1. Put the report text files and gold labels under `reproducibility_data/` or edit the config paths to your preferred location.
2. Run zero-shot on both cohorts first, because cross-dataset transfer depends on the opposite cohort's `zero_shot/all_models.pkl`.
3. Run the regex baseline if you want the rule-based comparison.
4. Run the LOO stage for the desired cohort and selector family.

Example commands:

```bash
obgynslms-run --config configs/reproducibility_minimal_mimic_long.json --stage zero-shot
obgynslms-run --config configs/reproducibility_minimal_indic_long.json --stage zero-shot
obgynslms-regex-baseline --config configs/reproducibility_minimal_mimic_long.json
obgynslms-run --config configs/reproducibility_minimal_mimic_long.json --stage loo
obgynslms-run --config configs/reproducibility_minimal_indic_long.json --stage loo
```

To inspect the run plan without model inference:

```bash
obgynslms-run --config configs/reproducibility_minimal_mimic_long.json --dry-run
```

## Reproduce Selection-Efficiency Experiments

`notebooks/fig_selection_efficiency.ipynb` does not generate the underlying data. It only reads precomputed CSV and JSON artifacts from:

- `results/<variant>/analysis/ping_pong_balanced_draw_stats_inference/`

For the default figure notebook, the notebook expects these variant roots:

- `results/mimic_streamlined_pipeline_small_models_upd_short`
- `results/mimic_streamlined_pipeline_small_models_upd_long`
- `results/mimic_streamlined_pipeline_small_models_indic_upd_short`
- `results/mimic_streamlined_pipeline_small_models_indic_upd_long`

The artifact generation chain is:

1. Run zero-shot first so each variant has `zero_shot/all_models.pkl`.
2. Run `scripts/run_balanced_draw_stats_inference.py`.
3. That script calls `scripts/run_balanced_draw_selection_stats.py` to materialize the per-target balanced-draw simulations.
4. `scripts/run_balanced_draw_stats_inference.py` then writes the summary CSVs used by the notebook.

The key files read by `fig_selection_efficiency.ipynb` are:

- `balanced_draw_stats_agg_k.csv`
- `balanced_draw_stats_primary_mean_over_k_comparisons.csv`
- `balanced_draw_stats_per_k_wilcoxon_comparisons.csv`
- `balanced_draw_stats_model_robustness_mean_over_k.csv`
- `balanced_draw_stats_regime_full_range_summary.csv`
- `inference_metadata.json`

Example regeneration commands:

```bash
python scripts/run_balanced_draw_stats_inference.py \
  --config configs/reproducibility_minimal_mimic_long.json \
  --variants long \
  --dataset mimic

python scripts/run_balanced_draw_stats_inference.py \
  --config configs/reproducibility_minimal_indic_long.json \
  --variants indic_long \
  --dataset our
```

For the full figure, run the same script for all required short/long variants after generating the corresponding zero-shot outputs. If you use different output roots than the canonical `results/...` variant names above, update the variant mapping in `scripts/run_balanced_draw_stats_inference.py` or adapt the notebook paths accordingly.

`notebooks/fig_selection_efficiency.py` is a standalone plotting export with hard-coded values; the canonical regeneration path for reproducibility use is the notebook plus the two scripts above.

## Verification

The repository includes synthetic minimal reproducibility tests:

```bash
pytest -q
```

These tests verify:

- relative config paths resolve correctly
- reproducibility dry-run works even before opposite-cohort cross-ping predictions exist
- the regex baseline CLI works with CSV gold labels

## Source Module Guide

Modules under `src/` are source code, not generated artifacts. I do not recommend adding any of the tracked `src/*.py` files to `.gitignore`; the current ignore rules should only cover generated caches such as `__pycache__`, `*.pyc`, and notebook checkpoints.

### Core workflow modules

| File | Purpose |
|---|---|
| `src/config_utils.py` | Resolves config paths, including reproducibility-friendly relative paths and dataset presets. |
| `src/data_loader.py` | Loads document corpora from a directory of `.txt` files into `{file, text}` records. |
| `src/evaluation_utils.py` | Aligns predictions with gold labels and computes per-model precision/recall/F1 summaries. |
| `src/experiment_logger.py` | Small shared logging helper used by model-based utilities. |
| `src/label_utils.py` | Normalizes raw outputs into canonical labels `{early, late, unrelated}`. |
| `src/loo_selector.py` | Main leave-one-out exemplar-selection and prompt-building logic for ICL experiments. |
| `src/model_runner.py` | Hugging Face inference wrapper for zero-shot and prompt-based classification runs. |
| `src/utils.py` | Checkpoint load/save helpers used during model inference. |
| `src/vote_entropy.py` | Core selector utilities for vote entropy, embeddings, and ping-style exemplar selection. |

### Analysis and reporting modules

| File | Purpose |
|---|---|
| `src/consolidated_loo_eval.py` | Builds consolidated leave-one-out evaluation tables for downstream reporting. |
| `src/cross_institutional_analysis.py` | Standalone cross-institution transfer analysis script that writes summary tables. |
| `src/eval.py` | Generic evaluation and paired bootstrap significance helpers used by the LOO pipeline. |
| `src/evalpairs.py` | Specialized significance utilities for comparing predefined medical-vs-general model pairs. |
| `src/evalping.py` | Specialized significance utilities for comparing one target method against other methods. |

### Auxiliary modules

| File | Purpose |
|---|---|
| `src/embeds.py` | Small embedding-similarity helper for inspecting nearest documents from a saved embedding file. |
| `src/text_normalizer.py` | Optional LLM-based report normalization / structured extraction utility; not part of the main reproducibility workflow. |
| `src/__init__.py` | Package marker. |

## Notes
- Result tables, figure exports, and cached experiment outputs should be generated into `results/` locally as needed.
- `README.md` is intended to be the complete top-level usage and reproducibility document.
