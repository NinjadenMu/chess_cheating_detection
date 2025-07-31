import math
import os
import pickle
from scipy.optimize import minimize
import sys

from data_processor import process_data
from engine import Engine
from loss import LossWrapper, MSELoss
from model import Model
import utils


config_path = f'{utils.get_root()}/config/train_config.json'

config = utils.load_config(config_path)

processed_data_path = config['data']['processed_data_path']

# if user has not already processed data, run the data processor
if not config['data']['data_is_processed']:
    engine = Engine(config['engine']['depth'], config['engine']['pv'])

    print('For large amounts of games, data processing may take several hours\n')
    process_data(config['data']['raw_data_path'], processed_data_path, engine)

data_files = os.listdir(processed_data_path)

super_game = []
# combine all games into one "super-game", training is done by fitting to this super-game
for data_file in data_files:
    if data_file.lower().endswith('.pkl'):
        processed_games = utils.load_processed_games(f'{processed_data_path}/{data_file}')

        for game in processed_games:
            super_game.extend(game[config['model']['opening_length_doubled']:])

if not os.path.exists(config['train']['log_dir']):
    os.makedirs(config['train']['log_dir'])

# initialize training hyperparameters
s_bounds = config['train']['s_bounds']
c_bounds = config['train']['c_bounds']

s_step = config['train']['s_step']
c_step = config['train']['c_step']

#initialize model parameters
s = s_bounds[0]
c = c_bounds[0]

# initialize model
model_config = config['model']
model = Model(0, model_config['ignore_threshold'], model_config['pv'])
model.train() # set model to training mode

loss_funcs = {}
projection_funcs = {}
if config['train']['ORF']:
    targets = model.calculate_mms(super_game, True)

    loss = MSELoss(targets, False)

    model = Model(0, model_config['ignore_threshold'], model_config['pv'])
    model.train() # set model to training mode

    loss_funcs['ORF'] = LossWrapper(model, loss, 'ORF', super_game)

if config['train']['MM']:
    targets = [model.calculate_mm(super_game, True)]

    loss = MSELoss(targets, False)

    model = Model(0, model_config['ignore_threshold'], model_config['pv'])
    model.train() # set model to training mode

    loss_funcs['MM'] = LossWrapper(model, loss, 'MM', super_game)

if config['train']['AE']:
    targets = [model.calculate_ae(super_game)]

    loss = MSELoss(targets, False)

    model = Model(0, model_config['ignore_threshold'], model_config['pv'])
    model.train() # set model to training mode

    loss_funcs['AE'] = LossWrapper(model, loss, 'AE', super_game)

if config['train']['MM_AE']:
    targets = [model.calculate_mm(super_game, True), model.calculate_ae(super_game)]

    loss = MSELoss(targets, True)
 
    model = Model(0, model_config['ignore_threshold'], model_config['pv'])
    model.train() # set model to training mode

    loss_funcs['MM_AE'] = LossWrapper(model, loss, 'MM_AE', super_game)

print(minimize(loss_funcs['ORF'], [0.33, 0.65], bounds = [[0.1, 0.5], [0.3, 0.8]], method='Nelder-Mead', options = {'xatol': 1e-5, 'disp': True}))
print(minimize(loss_funcs['MM'], [0.33, 0.65], bounds = [[0.1, 0.5], [0.3, 0.8]], method='Nelder-Mead', options = {'xatol': 1e-5, 'disp': True}))
print(minimize(loss_funcs['AE'], [0.33, 0.65], bounds = [[0.1, 0.5], [0.3, 0.8]], method='Nelder-Mead', options = {'xatol': 1e-5, 'disp': True}))
print(minimize(loss_funcs['MM_AE'], [0.33, 0.65], bounds = [[0.1, 0.5], [0.3, 0.8]], method='Nelder-Mead', options = {'xatol': 1e-5, 'disp': True}))
