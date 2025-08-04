import json
import os
from scipy.optimize import minimize

from data_processor import process_data
from engine import Engine
from loss import LossWrapper, MSELoss
from model import Model
import utils


config_dir = f'{utils.get_root()}/config'

config_path = f'{config_dir}/train_config.json'
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
            # add all non-opening moves to super game
            super_game.extend(game[config['model']['opening_length'] * 2:])

if not os.path.exists(config['train']['log_path']):
    os.makedirs(config['train']['log_path'])

if not os.path.exists(config['train']['save_path']):
    os.makedirs(config['train']['save_path'])

# initialize model
model_config = config['model']
opening_length = 0 # because opening moves were already removed from super_game, the model shouldn't ignore any moves
model = Model(opening_length, model_config['ignore_threshold'], model_config['pv'])

# build loss functions for each specified loss
loss_funcs = {}

if config['train']['ORF']:
    targets = model.calculate_mms(super_game, True)

    loss_func = MSELoss(targets, False)

    trainable_model = Model(opening_length, model_config['ignore_threshold'], model_config['pv'])
    trainable_model.train() # set model to training mode

    loss_funcs['ORF'] = LossWrapper(trainable_model, loss_func, 'ORF', super_game)

if config['train']['MM']:
    targets = [model.calculate_mm(super_game, True)]

    loss_func = MSELoss(targets, False)

    trainable_model = Model(opening_length, model_config['ignore_threshold'], model_config['pv'])
    trainable_model.train() # set model to training mode

    loss_funcs['MM'] = LossWrapper(trainable_model, loss_func, 'MM', super_game)

if config['train']['AE']:
    targets = [model.calculate_ae(super_game)]

    loss_func = MSELoss(targets, False)

    trainable_model = Model(opening_length, model_config['ignore_threshold'], model_config['pv'])
    trainable_model.train() # set model to training mode

    loss_funcs['AE'] = LossWrapper(trainable_model, loss_func, 'AE', super_game)

if config['train']['MM_AE']:
    targets = [model.calculate_mm(super_game, True), model.calculate_ae(super_game)]

    loss_func = MSELoss(targets, True)
 
    trainable_model = Model(opening_length, model_config['ignore_threshold'], model_config['pv'])
    trainable_model.train() # set model to training mode

    loss_funcs['MM_AE'] = LossWrapper(trainable_model, loss_func, 'MM_AE', super_game)

# set optimizer
optimizer = config['train']['optimizer']

optimizer_config_path = f'{config_dir}/optimizers/{optimizer}.json'
optimizer_config = utils.load_json(optimizer_config_path)

# set training hyperparameters
if optimizer == 'nelder-mead':
    start = optimizer_config['start']
    s_bounds = optimizer_config['s_bounds']
    c_bounds = optimizer_config['c_bounds']
    precision = optimizer_config['precision']

else:
    print('Currently, only the nelder-mead optimizer is implemented')

if optimizer == 'nelder-mead':
    print('Training Initiated...')

    for loss in loss_funcs.keys():
        print(f'  Minimizing {loss} loss...')
        
        params = minimize(loss_funcs[loss], start, bounds = [s_bounds, c_bounds], method='Nelder-Mead', options = {'xatol': precision})
        params = {
            's': params.x[0],
            'c': params.x[1]
                  }

        print(f'    Best Parameters: \n      s: {params['s']}, c: {params['c']}\n')

        with open(f'{config['train']['save_path']}/{loss}.json', 'w') as f:
            json.dump(params, f)