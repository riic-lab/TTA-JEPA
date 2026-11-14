from pathlib import Path
import yaml


def load_config(path):
    path = Path(path)
    config = yaml.safe_load(path.read_text())
    parent = config.pop("extends", None)
    if parent is None:
        return config
    base = load_config(path.parent / parent)
    for key, value in config.items():
        if isinstance(value, dict) and key != "model":
            base.setdefault(key, {}).update(value)
        else:
            base[key] = value
    return base
