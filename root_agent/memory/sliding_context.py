MAX_MESSAGES = 6

def trim_context(messages):
    system = [m for m in messages if m["role"] == "system"]
    recent = [m for m in messages if m["role"] != "system"][-MAX_MESSAGES:]
    return system + recent
