import csv
import json
import os
from datetime import datetime
from typing import Any, Dict

import yaml

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "outputs",
)


class ExperimentLogger:
    """Logs experiments to a JSON-lines file + a CSV summary table."""

    def __init__(self, name: str, run_dir: str = None):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if run_dir is None:
            run_dir = os.path.join(OUTPUT_DIR, name, ts)
        self.run_dir = run_dir
        os.makedirs(self.run_dir, exist_ok=True)
        self.log_file = os.path.join(self.run_dir, "run.jsonl")
        self._entries: Dict[str, Any] = {"run_id": ts}

    def log(self, key: str, value: Any) -> None:
        self._entries[key] = value

    def log_dict(self, d: Dict[str, Any]) -> None:
        self._entries.update(d)

    def save_config(self, cfg: Dict[str, Any], name: str = "config.yaml") -> None:
        with open(os.path.join(self.run_dir, name), "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

    def save_metrics(self, metrics: Dict[str, Any]) -> None:
        self._entries["metrics"] = metrics
        with open(os.path.join(self.run_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    def finalize(self) -> str:
        """Write the final run.jsonl artifact and return the run directory."""
        with open(self.log_file, "w") as f:
            json.dump(self._entries, f)
        return self.run_dir

    def append_row(self, row: Dict[str, Any], csv_path: str) -> None:
        """Append a row to a summary CSV (creates header on first write)."""
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
