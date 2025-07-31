from scipy.optimize import minimize

def f(x):
    return x[0] ** 2 + (x[1] - 1) ** 2

print(minimize(f, [1, 2]))

import pickle

with open('/Users/jaden/Dev/chess_cheating_detection/logs/log.pkl', 'rb') as f:
    f = pickle.load(f)
    print(f)