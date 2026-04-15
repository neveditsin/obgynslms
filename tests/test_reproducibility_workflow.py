from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "run_mimic_baseline_and_loo.py"
REGEX_EVAL = PROJECT_ROOT / "evaluate_mimic_regex_baseline.py"


def _write_docs(directory: Path, docs: dict[str, str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, text in docs.items():
        (directory / name).write_text(text, encoding="utf-8")


def _write_gold_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_reproducibility_dry_run_uses_relative_paths_and_allows_missing_cross_predictions(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    mimic_docs = tmp_path / "reproducibility_data" / "mimic_docs"
    indic_docs = tmp_path / "reproducibility_data" / "indic_docs"
    _write_docs(mimic_docs, {"m1.txt": "GA 8 weeks gestational sac is present."})
    _write_docs(indic_docs, {"i1.txt": "GA 16 weeks anatomy scan placenta anterior."})

    _write_gold_csv(
        tmp_path / "reproducibility_data" / "mimic_gold.csv",
        [{"file": "m1.txt", "label": "early active pregnancy", "rationale": "Early GA."}],
    )
    _write_gold_csv(
        tmp_path / "reproducibility_data" / "indic_gold.csv",
        [{"file": "i1.txt", "label": "late active pregnancy", "rationale": "Late GA."}],
    )

    config_path = config_dir / "reproducibility_minimal.json"
    config = {
        "paths": {
            "dataset": "mimic",
            "dataset_paths": {
                "mimic": {
                    "data_dir": "../reproducibility_data/mimic_docs",
                    "gold_labels": "../reproducibility_data/mimic_gold.csv",
                },
                "our": {
                    "data_dir": "../reproducibility_data/indic_docs",
                    "gold_labels": "../reproducibility_data/indic_gold.csv",
                    "base_predictions_path": "../outputs/indic_run/zero_shot/all_models.pkl",
                },
            },
            "output_root": "../outputs/mimic_run",
        },
        "models": ["mock/model"],
        "artifacts": {"write_samples_txt": False, "write_error_log": False},
        "document_sampling": {"enabled": False},
        "zero_shot": {"enabled": True, "output_subdir": "zero_shot"},
        "loo": {
            "enabled": True,
            "output_subdir": "loo",
            "run_selections": ["cross_ping"],
            "seeds": [0],
            "k_values": [2],
        },
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, str(RUNNER), "--config", str(config_path), "--dry-run"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
    expected_cross_predictions = (
        tmp_path / "outputs" / "indic_run" / "zero_shot" / "all_models.pkl"
    ).resolve()
    assert f"base_predictions={expected_cross_predictions}" in completed.stdout
    assert "[dry-run] Would run cross_ping/seed0_k2" in completed.stdout


def test_regex_baseline_cli_supports_relative_csv_gold_labels(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    docs_dir = tmp_path / "reproducibility_data" / "docs"
    output_root = tmp_path / "outputs" / "regex_eval"

    _write_docs(
        docs_dir,
        {
            "doc_early.txt": "GA 8 weeks 2 days with gestational sac and yolk sac.",
            "doc_late.txt": "GA 20 weeks anatomy scan with placenta anterior and fetal movements.",
            "doc_none.txt": "Postpartum pelvic ultrasound with no pregnancy identified.",
        },
    )
    _write_gold_csv(
        tmp_path / "reproducibility_data" / "gold_labels.csv",
        [
            {"file": "doc_early.txt", "label": "early active pregnancy"},
            {"file": "doc_late.txt", "label": "late active pregnancy"},
            {"file": "doc_none.txt", "label": "no active pregnancy"},
        ],
    )

    config = {
        "paths": {
            "dataset": "mimic",
            "data_dir": "../reproducibility_data/docs",
            "gold_labels": "../reproducibility_data/gold_labels.csv",
            "output_root": "../outputs/regex_eval",
        }
    }
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "regex_eval.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, str(REGEX_EVAL), "--config", str(config_path)],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout

    summary_path = output_root / "regex_baseline" / "regex_baseline_summary.json"
    predictions_path = output_root / "regex_baseline" / "regex_baseline_predictions.csv"
    assert summary_path.exists()
    assert predictions_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    predictions = pd.read_csv(predictions_path)

    assert summary["n_docs"] == 3
    assert "late active pregnancy" in set(predictions["pred_baseline"].tolist())
