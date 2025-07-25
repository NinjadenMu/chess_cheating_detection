import os
import pickle
from tqdm import tqdm

import chess.pgn


def process_data(data_path, output_path, engine):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    pgns = os.listdir(data_path)

    for id, pgn in tqdm(enumerate(pgns)):
        if pgn.lower().endswith('.pgn'):  # ignore hidden files
            pgn = open(f'{data_path}/{pgn}')

            games = [chess.pgn.read_game(pgn) for game in pgn]

            processed_games = []
            for game in games:
                if game != None:
                    processed_games.append(engine.process_game(game))

        with open(f'{output_path}/{id}.pkl', 'wb') as f:
            pickle.dump(processed_games, f)