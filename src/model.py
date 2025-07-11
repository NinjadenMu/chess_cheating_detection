import json
import math


class Model:
    def __init__(self, s, c, mm_sd_corrrectors, ae_sd_corrector, opening_length, ignore_threshold, pv):
        self.s = s
        self.c = c

        # sd_correctors adjusts projected standard deviations by a scalar
        # compensates for the sparse dependence between moves that violates the model's assumption of independence between moves
        self.mm_sd_correctors = mm_sd_corrrectors
        self.ae_sd_corrector = ae_sd_corrector

        self.opening_length = opening_length
        self.ignore_threshold = ignore_threshold

        self.pv = pv

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
    
    def calculate_probabilities(self, values, epsilon = 1e-6):
        # map raw centipawn evaluations to scaled deltas
        deltas = [self.value_to_delta(values[0], value) for value in values]

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
    
    # returns Boolean representing if {data_point} meets criteria of being statistically useful
    # player must have at least 3 legal moves, be out of book, position can't be too imbalanced
    def is_useful(self, data_point, move_number):
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
    
    # returns mm for all evaluated moves in a game
    def calculate_mms(self, processed_game):
        totals = [0 for i in range(self.pv)]
        for move_number, data_point in enumerate(processed_game):
            if data_point[-1] != -1: # checks if played move is in the top {pv} moves
                if self.is_useful(data_point, move_number):
                    totals[data_point[-1]] += 1

        return totals
    
    # returns model's projected distribution of mm for all evaluated moves in a game
    def project_mms(self, processed_game):
        means = [0 for rank in range(self.pv)]
        vars = [0 for rank in range(self.pv)]

        for move_number, data_point in enumerate(processed_game):
            values = data_point[:-1]

            if self.is_useful(data_point, move_number):
                probabilities = self.calculate_probabilities(values)

                for rank in range(self.pv):
                    means[rank] += probabilities[rank]
                    vars[rank] += probabilities[rank] * (1 - probabilities[rank])

        for rank in range(self.pv):
            vars[rank] = vars[rank] ** 0.5 * self.mm_sd_correctors[rank]
        
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