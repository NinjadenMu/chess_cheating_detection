import os
import pickle
import sys

from data_processor import process_data
from engine import Engine

from loss import MSELoss
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
            super_game.extend(game)

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
model = Model(model_config['opening_length'], model_config['ignore_threshold'], model_config['pv'])
model.train() # set model to training mode

# build loss functions and projection functions for each fitting method
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

# Use grid search to find parameters that minimize each loss
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
            loss['ORF'] = (loss_funcs['ORF'](projections))

        if config['train']['MM_AE']: # if mm_ae is a fitting method, use its projections for mm and ae individual fitting if needed
            projected_mm_mean, projected_mm_sd = model.project_mm(super_game, True)
            projected_ae_mean, projected_ae_sd = model.project_ae(super_game)
            loss['MM_AE'] = (loss_funcs['MM_AE']([projected_mm_mean, projected_ae_mean], [projected_mm_sd, projected_ae_sd]))

            if config['train']['MM']:
                loss['MM'] = (loss_funcs['MM']([projected_mm_mean]))

            if config['train']['AE']:
                loss['AE'] = (loss_funcs['AE']([projected_ae_mean]))

        else:
            if config['train']['MM']:
                projections = [model.project_mm(super_game, True)[0]]
                loss['MM'] = (loss_funcs['MM'](projections))

            if config['train']['AE']:
                projections = [model.project_ae(super_game)[0]]
                loss['AE'] = (loss_funcs['AE'](projections))

        losses[(s, c)] = loss

        c += c_step

        with open(f"{config['train']['log_dir']}/log.pkl", 'wb') as f:
            pickle.dump(losses, f)

    s += s_step

