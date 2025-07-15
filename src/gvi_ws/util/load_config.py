import yaml

def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)