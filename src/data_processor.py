import os
import pickle
from tqdm import tqdm

import chess.pgn


"""
Reads all PGNs in {data_path} and saves engine analyses as pickle files

Each pickle file is a list containing the analysis of each game in a corresponding PGN

If a specific player is provided in {player_to_track}, 
{id}_w.pkl will store the analyses of their white games and {id}_b.pkl will store the analyses of their black games
"""
def process_data(data_path, output_path, engine, player_to_track = None):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    pgns = sorted(os.listdir(data_path))

    for id, pgn in tqdm(enumerate(pgns)):
        if pgn.lower().endswith('.pgn'):  # ignore hidden files
            pgn = open(f'{data_path}/{pgn}')

            games = [chess.pgn.read_game(pgn) for game in pgn]

            if player_to_track != None:
                processed_white_games = []
                processed_black_games = []
                for game in games:
                    if game != None:
                        if game.headers['White'] == player_to_track:
                            processed_white_games.append(engine.process_game(game))

                        elif game.headers['Black'] == player_to_track:
                            processed_black_games.append(engine.process_game(game))

                with open(f'{output_path}/{id}_w.pkl', 'wb') as f:
                    pickle.dump(processed_white_games, f)

                with open(f'{output_path}/{id}_b.pkl', 'wb') as f:
                    pickle.dump(processed_black_games, f)

            else:    
                processed_games = []
                for game in games:
                    if game != None:
                        processed_games.append(engine.process_game(game))

                with open(f'{output_path}/{id}.pkl', 'wb') as f:
                    pickle.dump(processed_games, f)