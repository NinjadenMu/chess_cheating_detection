import json
import pickle


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)

    return config

def load_processed_games(path):
    with open(path, 'rb') as f:
        processed_games = pickle.load(f)

    return processed_games