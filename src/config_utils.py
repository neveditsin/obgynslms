from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def resolve_path(value: Any, *, base_dir: Optional[Path] = None) -> Optional[Path]:
    """Resolve config paths relative to the config file directory."""

    if value is None:
        return None

    raw = str(value).strip()
    if not raw:
        return None

    path = Path(raw).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path.resolve()


def resolve_dataset_paths(
    paths_cfg: Dict[str, Any],
    *,
    base_dir: Optional[Path] = None,
) -> Tuple[Path, Optional[Path], Dict[str, Any]]:
    """Resolve dataset paths from config, normalizing nested dataset presets too."""

    resolved = dict(paths_cfg)
    dataset = str(resolved.get("dataset", "mimic")).strip().lower()
    resolved["dataset"] = dataset

    if resolved.get("output_root"):
        resolved["output_root"] = str(resolve_path(resolved["output_root"], base_dir=base_dir))
    if resolved.get("base_predictions_path"):
        resolved["base_predictions_path"] = str(
            resolve_path(resolved["base_predictions_path"], base_dir=base_dir)
        )

    data_dir_value = resolved.get("data_dir")
    gold_labels_value = resolved.get("gold_labels")

    dataset_paths = resolved.get("dataset_paths")
    if dataset_paths is not None:
        if not isinstance(dataset_paths, dict):
            raise ValueError("Config.paths.dataset_paths must be an object if provided.")

        normalized_dataset_paths: Dict[str, Dict[str, Any]] = {}
        for raw_name, raw_entry in dataset_paths.items():
            if not isinstance(raw_entry, dict):
                raise ValueError(
                    f"Config.paths.dataset_paths['{raw_name}'] must be an object with "
                    "'data_dir' and optional 'gold_labels'."
                )
            entry = dict(raw_entry)
            for key in ("data_dir", "gold_labels", "output_root", "base_predictions_path"):
                if entry.get(key):
                    entry[key] = str(resolve_path(entry[key], base_dir=base_dir))
            normalized_dataset_paths[str(raw_name).strip().lower()] = entry

        if dataset not in normalized_dataset_paths:
            available = ", ".join(sorted(str(k) for k in normalized_dataset_paths.keys()))
            raise ValueError(
                f"Config.paths.dataset='{dataset}' not found in dataset_paths. "
                f"Available: {available}"
            )

        resolved["dataset_paths"] = normalized_dataset_paths
        dataset_entry = normalized_dataset_paths.get(dataset, {})
        data_dir_value = dataset_entry.get("data_dir", data_dir_value)
        gold_labels_value = dataset_entry.get("gold_labels", gold_labels_value)

    if not data_dir_value:
        raise ValueError(
            "Could not resolve data directory. Set Config.paths.data_dir "
            "or Config.paths.dataset_paths.<dataset>.data_dir."
        )

    data_dir = resolve_path(data_dir_value, base_dir=base_dir)
    gold_labels_path = (
        resolve_path(gold_labels_value, base_dir=base_dir) if gold_labels_value else None
    )

    if data_dir is None:
        raise ValueError("Resolved data directory is empty.")

    resolved["data_dir"] = str(data_dir)
    if gold_labels_path is not None:
        resolved["gold_labels"] = str(gold_labels_path)

    return data_dir, gold_labels_path, resolved
