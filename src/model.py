import json
import math

import utils


class Model:
    def __init__(self, opening_length, ignore_threshold, pv):
        self.opening_length = opening_length
        self.ignore_threshold = ignore_threshold

        self.pv = pv   

    # loads fitted parameters for eval mode
    def load(self, model_params):
        if not isinstance(model_params, dict):
            model_params = utils.load_json(model_params)

        self.s = model_params['s']
        self.c = model_params['c']

        # sd_correctors adjusts projected standard deviations by a scalar
        # compensates for the sparse dependence between moves that violates the model's assumption of independence between moves
        self.mm_sd_correctors = model_params['mm_sd_corrrectors']
        self.ae_sd_corrector = model_params['ae_sd_corrector']

    # configures model to training mode
    def train(self):
        self.s = None
        self.c = None

        self.mm_sd_correctors = [1 for rank in range(20)]
        self.ae_sd_corrector = 1

    def save(self, save_path = None):
        model_params = {
            's': self.s,
            'c': self.c,
            'mm_sd_correctors': self.mm_sd_correctors,
            'ae_sd_corrector': self.ae_sd_corrector
        }

        if save_path != None:
            with open(save_path, 'w') as f:
                json.dump(model_params, f)

        return model_params

    # scales errors according to magnitude of advantage (small mistakes when one side has a large advantage are less important)
    def scaling_law(self, value):
        if value >= 0:
            return math.log(1 + value / 100)
        
        return -1 * math.log(1 - value / 100)
    
    def value_to_delta(self, eval_best, eval):
        return abs(self.scaling_law(eval_best) - self.scaling_law(eval))
    
    def delta_to_proxy(self, delta):
        return math.exp(-1 * (delta / self.s) ** self.c)
    
    def proxy_to_probability(self, proxy, probability_best):
        return probability_best ** (1 / proxy)
    
    # returns final projected probabilities of each move given their evaluations
    def calculate_probabilities(self, values, epsilon = 1e-6):
        # map raw centipawn evaluations to scaled deltas
        deltas = [self.value_to_delta(values[0], value) for value in values[:self.pv]]

        # map deltas to proxies
        proxies = [self.delta_to_proxy(delta) for delta in deltas]

        # use binary search to numerically find probabilities that satisfy log-log relationship with proxies
        low = 0
        high = 1

        while high - low > epsilon:
            probability_best = (low + high) / 2

            if sum(self.proxy_to_probability(proxy, probability_best) for proxy in proxies) < 1:
                low = probability_best

            else:
                high = probability_best

        probability_best = (low + high) / 2

        probabilities = [self.proxy_to_probability(proxy, probability_best) for proxy in proxies]
        # if less than {pv} moves were evaluated, assign 0 probabiity to "phantom moves" so model always processes {pv} moves
        probabilities.extend([0 for i in range(self.pv - len(probabilities))])

        return probabilities
    
    # returns Boolean representing if {data_point} meets criteria for being statistically useful
    def is_useful(self, data_point, move_number):
        # player must have at least 3 legal moves, be out of book, position can't be too imbalanced
        if len(data_point) > 3 and move_number >= self.opening_length and abs(data_point[0]) < self.ignore_threshold:
            return True
        
        return False
    
    # returns frequency of the #{rank} best move played in a game (move match) 
    def calculate_mm(self, processed_game, proportion = False, rank = 0):
        total = 0
        num_moves = 1
        for move_number, data_point in enumerate(processed_game):
            if self.is_useful(data_point, move_number):
                num_moves += 1

                if data_point[-1] == rank:
                    total += 1

        if proportion:
            return total / num_moves
        
        return total
    
    # returns model's projected distribution of move match (mm) for the #{rank} best move in a game
    def project_mm(self, processed_game, proportion = False, rank = 0):
        mean = 0
        var = 0
        num_moves = 1
        for move_number, data_point in enumerate(processed_game):
            values = data_point[:-1]

            if self.is_useful(data_point, move_number):
                num_moves += 1
                
                probability = self.calculate_probabilities(values)[rank]

                mean += probability
                var += probability * (1 - probability) # thanks Bernoulli :)

        if proportion:
            return mean / num_moves, var ** 0.5 / num_moves * self.mm_sd_correctors[rank]

        return mean, var ** 0.5 * self.mm_sd_correctors[rank]
    
    # returns mm for all ranked moves in a game
    def calculate_mms(self, processed_game, proportion = False):
        totals = [0 for rank in range(self.pv)]
        num_moves = 1
        for move_number, data_point in enumerate(processed_game):
            if data_point[-1] != -1: # checks if played move is in the top {pv} moves
                if self.is_useful(data_point, move_number):
                    num_moves += 1

                    totals[data_point[-1]] += 1

        if proportion:
            for rank in range(self.pv):
                totals[rank] /= num_moves

        return totals
    
    # returns model's projected distribution of mm for all ranked moves in a game
    def project_mms(self, processed_game, proportion = False):
        means = [0 for rank in range(self.pv)]
        vars = [0 for rank in range(self.pv)]
        num_moves = 1
        for move_number, data_point in enumerate(processed_game):
            values = data_point[:-1]

            if self.is_useful(data_point, move_number):
                num_moves += 1

                probabilities = self.calculate_probabilities(values)

                for rank in range(self.pv):
                    means[rank] += probabilities[rank]
                    vars[rank] += probabilities[rank] * (1 - probabilities[rank])

        for rank in range(self.pv):
            vars[rank] = vars[rank] ** 0.5 * self.mm_sd_correctors[rank]

            if proportion:
                means[rank] /= num_moves
                vars[rank] /= num_moves
        
        return means, vars
    
    # returns average error (ae) of a game (essentially a scaled average centipawn loss)
    def calculate_ae(self, processed_game):
        error = 0
        num_moves = 1
        for move_number, data_point in enumerate(processed_game):
            if data_point[-1] != -1:
                if self.is_useful(data_point, move_number):
                    num_moves += 1

                    error += self.value_to_delta(data_point[0], data_point[data_point[-1]])

        return error / num_moves
    
    # returns distribution of model's projected ae of a game
    def project_ae(self, processed_game):
        mean = 0
        var = 0
        num_moves = 1
        for move_number, data_point in enumerate(processed_game):
            values = data_point[:-1]

            if data_point[-1] != -1:
                if self.is_useful(data_point, move_number):
                    num_moves += 1

                    probabilities = self.calculate_probabilities(values)

                    move_error = 0
                    for rank in range(1, len(values)):
                        probability = probabilities[rank]

                        delta = self.value_to_delta(data_point[0], data_point[rank])

                        move_error += probability * delta

                        var += probability * delta ** 2

                    var -= move_error ** 2
                    mean += move_error

        return mean / num_moves, var ** 0.5 / num_moves * self.ae_sd_corrector
    
if __name__ == '__main__':
    model = Model(0.31, 0.65, [1.14 for i in range(20)], 1.4, 5, 350, 20)

    import pickle
    games = []
    with open('/Users/jaden/Dev/chess_cheating_detection/processed_eval_data/1.pkl', 'rb') as f:
        games.append(pickle.load(f)[0])
    with open('/Users/jaden/Dev/chess_cheating_detection/processed_eval_data/2.pkl', 'rb') as f:
        games.append(pickle.load(f)[0])

    split_games = [[] for i in range(4)]
    for i in range(len(games[0])):
        if i % 2 == 0:
            split_games[0].append(games[0][i])
        else:
            split_games[1].append(games[0][i])

    for i in range(len(games[1])):
        if i % 2 == 0:
            split_games[2].append(games[1][i])
        else:
            split_games[3].append(games[1][i])
    a = split_games[0].extend(split_games[2])
    mms = [model.calculate_mm(game) for game in split_games]
    mm_distributions = [model.project_mm(game) for game in split_games]
    aes = [model.calculate_ae(game) for game in split_games]
    ae_distributions = [model.project_ae(game) for game in split_games]

    print(mms)
    print(mm_distributions)
    print(aes)
    print(ae_distributions)

    print(model.calculate_ae(a), model.project_ae(a))
    print(model.calculate_mm(a), model.project_mm(a))

