# import json
# import time
# from pathlib import Path

# LOG_FILE = Path("token_log.jsonl")

# def log_event(event: str, payload: dict):
#     payload["event"] = event
#     payload["ts"] = time.time()
#     LOG_FILE.parent.mkdir(exist_ok=True)
#     with open(LOG_FILE, "a", encoding="utf-8") as f:
#         f.write(json.dumps(payload) + "\n")


import json, time
from pathlib import Path

LOG_FILE = Path("token_log.jsonl")

def log_event(event: str, payload: dict):
    payload = dict(payload)  # avoid mappingproxy
    payload["event"] = event
    payload["ts"] = time.time()

    LOG_FILE.parent.mkdir(exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")
