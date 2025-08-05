import json
from pathlib import Path
import pickle


def get_root():
    return Path(__file__).resolve().parents[1]

def load_json(json_path):
    with open(json_path, 'r') as f:
        object = json.load(f)

    return object

def load_config(config_path):
    root = get_root()

    config = load_json(config_path)

    # make all relative paths in config absolute
    for section in config.keys():
        for field in config[section].keys():
            if 'path' in field:
                if config[section][field][0] != '/':
                    config[section][field] = f'{root}/{config[section][field]}'

    return config

def load_processed_games(path):
    with open(path, 'rb') as f:
        processed_games = pickle.load(f)

    return processed_games