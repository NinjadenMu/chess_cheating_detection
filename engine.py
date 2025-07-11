from stockfish import Stockfish

class Engine:
    def __init__(self, depth, pv):
        self.engine = Stockfish(depth = depth)
        self.pv = pv  # Stockfish will evaluate top {pv} moves 

    def get_values(self, fen):
        self.engine.set_fen_position(fen)

        return self.engine.get_top_moves(self.pv)
    
    def process_game(self, game):
        processed_game = []

        board = game.board()
        for move in game.mainline_moves():
            values = self.get_values(board.fen())

            # each data_point will contain evaluations of top {pv} move-options followed by the index of the played move
            data_point = []
            played_move_index = -1
            for i, value in enumerate(values):
                # ignores all moves evaluating to forced mate to ensure model only receives numerical evals
                if value['Centipawn'] != None:
                    if value['Move'] == str(move):
                        played_move_index = i

                    data_point.append(value['Centipawn'])

                else:
                    break

            data_point.append(played_move_index)

            processed_game.append(data_point)

            board.push(move)

        return processed_game
