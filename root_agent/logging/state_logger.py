from pathlib import Path
import json
from datetime import datetime

STATE_LOG_DIR = Path("state_logs")
STATE_LOG_DIR.mkdir(exist_ok=True)

def log_state(session_id: str, state: dict):
    file = STATE_LOG_DIR / f"{session_id}.jsonl"

    with open(file, "a", encoding="utf-8") as f:
        record = {
            "time": datetime.utcnow().isoformat(),
            "state": state,
        }
        f.write(json.dumps(record) + "\n")
