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

"""# build loss functions and projection functions for each fitting method
loss_funcs = {}
projection_funcs = {}
if config['train']['ORF']:
    targets = model.calculate_mms(super_game, True)

    loss = MSELoss(targets, False)
    loss_funcs['ORF'] = loss

if config['train']['MM']:
    targets = [model.calculate_mm(super_game, True)]

    loss = MSELoss(targets, False)
    loss_funcs['MM'] = loss

if config['train']['AE']:
    targets = [model.calculate_ae(super_game)]

    loss = MSELoss(targets, False)
    loss_funcs['AE'] = loss

if config['train']['MM_AE']:
    targets = [model.calculate_mm(super_game, True), model.calculate_ae(super_game)]

    loss = MSELoss(targets, True)
    loss_funcs['MM_AE'] = loss

# Use grid search to find losses for each combo of parameters
losses = {}
while s < s_bounds[1] + 1e-5: # 1e-5 accounts for floating point errors
    sys.stdout.write(f'\rProgress: {round((s - s_bounds[0]) / (s_bounds[1] - s_bounds[0]) * 100)}% (For large amounts of games, this may take several hours)')
    sys.stdout.flush()
    while c < c_bounds[1] + 1e-5:
        model.s = s
        model.c = c

        loss = {}
        if config['train']['ORF']:
            projections = model.project_mms(super_game, True)[0]
            loss['ORF'] = loss_funcs['ORF'](projections)

        if config['train']['MM_AE']: # if mm_ae is a fitting method, use its projections for mm and ae individual fitting if needed
            projected_mm_mean, projected_mm_sd = model.project_mm(super_game, True)
            projected_ae_mean, projected_ae_sd = model.project_ae(super_game)
            loss['MM_AE'] = loss_funcs['MM_AE']([projected_mm_mean, projected_ae_mean], [projected_mm_sd, projected_ae_sd])

            if config['train']['MM']:
                loss['MM'] = loss_funcs['MM']([projected_mm_mean])

            if config['train']['AE']:
                loss['AE'] = loss_funcs['AE']([projected_ae_mean])

        else:
            if config['train']['MM']:
                projections = [model.project_mm(super_game, True)[0]]
                loss['MM'] = loss_funcs['MM'](projections)

            if config['train']['AE']:
                projections = [model.project_ae(super_game)[0]]
                loss['AE'] = loss_funcs['AE'](projections)

        losses[(s, c)] = loss

        c += c_step

        with open(f"{config['train']['log_dir']}/log.pkl", 'wb') as f:
            pickle.dump(losses, f)

    s += s_step
    c = c_bounds[0]

# Iterate through all combos of parameters to find combos with lowest losses
best_losses = {}
for fitting_method in loss_funcs.keys():
    best_losses[fitting_method] = [math.inf, (None, None)]

for parameters in losses.keys():
    for fitting_method in losses[parameters].keys():
        loss = losses[parameters][fitting_method]
        if loss < best_losses[fitting_method][0]:
            best_losses[fitting_method] = [loss, parameters]
            
print(best_losses)"""

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
