from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run_mimic_baseline_and_loo import (
    _selection_overrides,
    _selection_prompt_kwargs,
    _validate_selection_names,
)
from src.loo_selector import LOOResources, LeaveOneOutPromptBuilder


def _build_resources() -> LOOResources:
    vote_table = pd.DataFrame(
        [
            {"file": "doc1", "top_label": "early", "vote_entropy": 0.05},
            {"file": "doc2", "top_label": "late", "vote_entropy": 0.15},
            {"file": "doc3", "top_label": "unrelated", "vote_entropy": 0.25},
        ]
    )
    text_lookup = {
        "doc1": "Target report text.",
        "doc2": "Example report for a late pregnancy.",
        "doc3": "Example report with no active pregnancy.",
    }
    gold_df = pd.DataFrame(
        [
            {"file": "doc1", "label": "early", "rationale": "Gestational sac is present."},
            {"file": "doc2", "label": "late", "rationale": "Second-trimester anatomy is described."},
            {"file": "doc3", "label": "unrelated", "rationale": "The report documents a resolved pregnancy."},
        ]
    )
    return LOOResources(
        vote_table=vote_table,
        text_lookup=text_lookup,
        gold_df=gold_df,
    )


def test_random_ra_prompt_matches_random_selection_and_adds_rationales() -> None:
    resources = _build_resources()
    doc = {"file": "doc1", "text": resources.text_lookup["doc1"]}

    random_builder = LeaveOneOutPromptBuilder(
        resources,
        selection="random",
        selection_kwargs={"k": 1, "seed": 7},
    )
    random_ra_builder = LeaveOneOutPromptBuilder(
        resources,
        selection="random_ra",
        selection_kwargs={"k": 1, "seed": 7},
    )

    random_prompt = random_builder.build(doc)
    random_ra_prompt = random_ra_builder.build(doc)

    assert "Example report for a late pregnancy." in random_prompt
    assert "Example report with no active pregnancy." in random_prompt
    assert "Example report for a late pregnancy." in random_ra_prompt
    assert "Example report with no active pregnancy." in random_ra_prompt
    assert "Rationale:" not in random_prompt
    assert "Rationale: Second-trimester anatomy is described." in random_ra_prompt
    assert "Rationale: The report documents a resolved pregnancy." in random_ra_prompt


def test_random_ra_inherits_random_prompt_kwargs_and_selection_overrides() -> None:
    loo_cfg = {
        "prompt_kwargs": {"input_tag": "Report"},
        "prompt_kwargs_by_selection": {
            "random": {"include_group_headers": True},
            "random_ra": {"rationale_tag": "Why"},
        },
        "selection_overrides": {
            "random": {"simulate_reveal": False},
            "random_ra": {"seed": 11},
        },
    }

    assert _selection_prompt_kwargs(loo_cfg, "random_ra") == {
        "input_tag": "Report",
        "include_group_headers": True,
        "rationale_tag": "Why",
    }
    assert _selection_overrides(loo_cfg, "random_ra") == {
        "simulate_reveal": False,
        "seed": 11,
    }


def test_cross_ping_ra_inherits_ping_prompt_kwargs_and_selection_overrides() -> None:
    loo_cfg = {
        "prompt_kwargs": {"input_tag": "Report"},
        "prompt_kwargs_by_selection": {
            "ping": {"include_group_headers": True},
            "cross_ping_ra": {"rationale_tag": "Why"},
        },
        "selection_overrides": {
            "ping": {"score": "harmonic", "use_random_pong": False},
            "cross_ping_ra": {"seed": 11},
        },
    }

    assert _selection_prompt_kwargs(loo_cfg, "cross_ping_ra") == {
        "input_tag": "Report",
        "include_group_headers": True,
        "rationale_tag": "Why",
    }
    assert _selection_overrides(loo_cfg, "cross_ping_ra") == {
        "score": "harmonic",
        "use_random_pong": False,
        "seed": 11,
    }


def test_random_ra_is_a_valid_selection_name() -> None:
    _validate_selection_names(
        ["random", "random_ra", "ping", "ping_ra", "cross_ping", "cross_ping_ra"]
    )


def test_cross_ping_ra_keeps_pool_when_target_file_id_matches_source_id() -> None:
    resources = LOOResources(
        vote_table=pd.DataFrame(
            [
                {"file": "shared.txt", "top_label": "early", "vote_entropy": 0.05},
            ]
        ),
        text_lookup={"shared.txt": "Cross-dataset exemplar text."},
        gold_df=pd.DataFrame(
            [
                {"file": "shared.txt", "label": "early", "rationale": "Cross rationale."},
            ]
        ),
    )

    builder = LeaveOneOutPromptBuilder(
        resources,
        selection="cross_ping_ra",
        selection_kwargs={"k": 1, "seed": 7, "use_random_pong": False},
        exclude_target_from_pool=False,
    )

    prompt = builder.build({"file": "shared.txt", "text": "Target report text."})

    assert "Cross-dataset exemplar text." in prompt
    assert "Rationale: Cross rationale." in prompt
    assert "NOW YOUR TURN" in prompt
