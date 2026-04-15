from pathlib import Path
from typing import List, Dict
import logging


def load_documents(directory: str) -> List[Dict[str, str]]:
    """Load all .txt files in a directory into a list of dicts.

    Each dict has keys ``file`` and ``text``.
    """
    docs: List[Dict[str, str]] = []
    path = Path(directory)
    if not path.exists():
        logging.error("Data directory %s does not exist", directory)
        return docs

    for file in sorted(path.glob("*.txt")):
        try:
            text = file.read_text(encoding="utf-8")
        except Exception as exc:
            logging.warning("Could not read %s: %s", file, exc)
            continue
        docs.append({"file": file.name, "text": text})
    logging.info("Loaded %d documents from %s", len(docs), directory)
    return docs
