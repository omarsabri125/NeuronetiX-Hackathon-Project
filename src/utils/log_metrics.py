import json
from pathlib import Path
from datetime import datetime, timezone
import uuid

BASE_DIR = Path(__file__).resolve().parents[2]
METRICS_FILE = BASE_DIR / "metrics.jsonl"

def log_metrics(model_name, smoteenn_applied, results):

    record = {
        "run_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "smoteenn_applied": smoteenn_applied,
        "results": results
    }

    with open(METRICS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")