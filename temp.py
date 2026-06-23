import os
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

OUTPUT_PATH = "final_load.parquet"

# Backup directory and backup frequency
BACKUP_DIR = "backups"
BACKUP_EVERY_N_CALLS = 50   # write a backup every N API calls

def save_backup(records: List[Dict[str, Any]], label: str) -> None:
    """Write a backup CSV for results (including metadata_json)."""
    if not records:
        return

    os.makedirs(BACKUP_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{label}_{ts}"

    backup_main = os.path.join(BACKUP_DIR, f"{base}.parquet")
    pd.DataFrame(records).to_parquet(backup_main)

    print(f"[backup] Saved backup to {backup_main}")


def main():
    all_records: List[Dict[str, Any]] = []
    # Final save
    if all_records:
        pd.DataFrame(all_records).to_parquet(
            OUTPUT_PATH,
        )
        print(f"\nSaved final results (including metadata_json) to {OUTPUT_PATH}")
    else:
        print("\nNo records were generated.")


if __name__ == "__main__":
    main()

    