import json
from pathlib import Path

def load_default_config():
    default_path = Path(__file__).parent / "default_config.json"
    with open(default_path) as f:
        return json.load(f)

def load_user_config(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        return json.load(f)