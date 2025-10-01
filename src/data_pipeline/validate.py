from __future__ import annotations
from pathlib import Path
from .config import RAW_DIR, FILENAMES
from .io_utils import ensure_materialized

def validate_raw():
    errors = []
    for k, fname in FILENAMES.items():
        p = Path(RAW_DIR) / fname
        try:
            ensure_materialized(p)
        except Exception as e:
            errors.append(f"{fname}: {e}")
    if errors:
        raise SystemExit("Raw checks failed:\n- " + "\n- ".join(errors))
    print("All raw files present & materialized ✅")
