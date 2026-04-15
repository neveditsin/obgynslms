from pathlib import Path
import pandas as pd
from typing import Optional


def load_checkpoint(path: str) -> pd.DataFrame:
    """Load checkpoint if it exists."""
    p = Path(path)
    if p.exists():
        return pd.read_pickle(p)
    return pd.DataFrame(columns=["file", "model", "raw_output"])


def save_checkpoint(df: pd.DataFrame, path: str) -> None:
    """Save DataFrame to a pickle file, creating parent directories."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(p)
