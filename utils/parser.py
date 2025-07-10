# utils/parser.py
import yaml
import argparse

def fix_backslashes_in_paths(config):
    for key, value in config.items():
        if isinstance(value, str):  # Nur Strings bearbeiten
            # Ersetze Backslashes durch Slashes
            config[key] = value.replace("\\", "/")
        elif isinstance(value, dict):  # Wenn es ein verschachteltes Dictionary ist
            fix_backslashes_in_paths(value)
        elif isinstance(value, list):  # Wenn es eine Liste ist
            for i in range(len(value)):
                if isinstance(value[i], str):
                    value[i] = value[i].replace("\\", "/")
    return config

def load_config():
    parser = argparse.ArgumentParser(description="Run pipeline with config.yaml")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config.yaml file"
    )
    args = parser.parse_args()

    with open(args.config, "r") as stream:
        config = yaml.safe_load(stream)

    config = fix_backslashes_in_paths(config)

    return config