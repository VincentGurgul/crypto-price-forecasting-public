''' Hyperparameter optimisation for crypto price forecasting with Random Forest. '''

import sys
sys.path += ['../../', '../../../']

import numpy as np
import pandas as pd
import json
import optuna
import wandb

from argparse import ArgumentParser
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from functions import *

from utils.wrappers import timeit, telegram_notify


@timeit
@telegram_notify
def hyperparameter_opt_rf():
    
    # parse cross validation folds and optuna iterations as args
    parser = ArgumentParser(
        description='Crypto price forecasting with Random Forest'
    )
    parser.add_argument(
        '-f',
        '--folds',
        type=int,
        default=7,
        help='number of folds for each time series cross validation, default = 7',
    )
    parser.add_argument(
        '-i',
        '--iter',
        type=int,
        default=50,
        help='number of trial runs per optuna run, i.e. per model, default = 50',
    )
    args = parser.parse_args()

    # Print args to results file
    with open('optuna_results.txt', 'w') as f:
        f.write(f'''Args:
    Optimising profit based on time series cross validation with {args.folds*2} folds.
    Optuna running {args.iter} trial runs per model.\n\n''')
    
    # Set model configuration that isn't up for optimisation
    base_config = {
        'n_jobs': -1,
        'random_state': 42,
    }
    
    for coin in ('btc', 'eth'):
        
        # Only BTC for this trial run
        if coin == 'eth':
            continue
        
        price_data = pd.read_parquet(f'../../../2_data_processing/numeric_data/{coin}_price_data.parquet.gzip').iloc[:,0]
        
        with open('optuna_results.txt', 'a') as f:
            f.write(f'----------------\nCoin: {coin.upper()}\n----------------')
        
        # load data
        data_path = f'../../../4_eda/{coin}_stationary_data_lagged.parquet.gzip'
        data = pd.read_parquet(data_path)
        data.columns = data.columns.map('_'.join)
        data = (data.fillna(method='ffill')
                    .fillna(0)
                    .replace([np.inf, -np.inf], 0))
        target_path = f'../../../2_data_processing/numeric_data/{coin}_targets.parquet.gzip'
        targets = pd.read_parquet(target_path)
        
        # select only the variables with a statistically significant causal relationship
        with open(f'../../../4_eda/{coin}_causality/{coin}_price_log_difference_causality.txt') as f:
            price_log_difference_vars = f.read().splitlines()
        X = data[price_log_difference_vars]
        
        y = targets[f'{coin}_price_log_difference']

        for problem in ('regression', 'classification'):
            
            # specify model and configuration for cross validation
            if problem == 'regression':
                model = RandomForestRegressor
                # Set range for hyperparameter optimisation
                iter_config = {
                    'name': 'n_estimators',
                    'low': 550,
                    'high': 650,
                }
                split_config = {
                    'name': 'min_samples_split',
                    'low': 20,
                    'high': 21,
                }
                leaf_config = {
                    'name': 'min_samples_leaf',
                    'low': 8,
                    'high': 8,
                }
                crit_config = {
                    'name': 'criterion',
                    'choices': ['squared_error', 'friedman_mse'],
                }
                cv_config = {
                    'add_constant_sliding_window': False,
                    'cutoff_tuning': False,
                }
            elif problem == 'classification':
                model = RandomForestClassifier
                # Set range for hyperparameter optimisation
                iter_config = {
                    'name': 'n_estimators',
                    'low': 115,
                    'high': 145,
                }
                split_config = {
                    'name': 'min_samples_split',
                    'low': 15,
                    'high': 17,
                }
                leaf_config = {
                    'name': 'min_samples_leaf',
                    'low': 5,
                    'high': 6,
                }
                crit_config = {
                    'name': 'criterion',
                    'choices': ['gini'],
                }
                weight_config = {
                    'name': 'minority_class_weight',
                    'low': 3.55,
                    'high': 3.9,
                }
                cv_config = {
                    'add_constant_sliding_window': False,
                    'cutoff_tuning': True,
                    'tuning_method': 'accuracy',
                }
            
            print(f'\nRunning optimisation for {coin} price movement {problem}...\n')
            
            # Define the objective function for optuna
            STUDY_NAME = f'{coin}_movement_{problem}'
            def objective(trial):
                
                model_config = {
                    'criterion': trial.suggest_categorical(**crit_config),
                    'n_estimators': trial.suggest_int(**iter_config),
                    'min_samples_split': trial.suggest_int(**split_config),
                    'min_samples_leaf': trial.suggest_int(**leaf_config),
                }
                if problem == 'classification':
                    minority_class_weight = trial.suggest_float(**weight_config)
                    model_config['class_weight'] = {0: 1, 1: minority_class_weight}
                    
                classifier_obj = model(**model_config, **base_config)
                
                log_config = {
                    'trial_number': trial.number,
                    **model_config,
                    **cv_config,
                }
                try:
                    log_config['minority_class_weight'] = minority_class_weight
                except:
                    pass
                    
                wandb.init(
                    project='Crypto RF',
                    config=log_config,
                    group=STUDY_NAME,
                    reinit=True,
                )
                result = crossvalidate_movement(
                    classifier_obj,
                    problem,
                    X,
                    y,
                    price_data,
                    args.folds,
                    wandb_log=True,
                    **cv_config,
                )
                wandb.log(data={'avg_profit': result[3]})
                wandb.finish()
                return result[3]

            # Run optuna study
            study = optuna.create_study(
                direction='maximize',
                study_name=STUDY_NAME,
            )
            study.optimize(objective, n_trials=args.iter)
                        
            # Log optuna summary plots to W&B
            try:
                param_importances = optuna.visualization.plot_param_importances(study)
                optimization_history = optuna.visualization.plot_optimization_history(study)
                wandb.init(
                    project='Crypto RF',
                    group=STUDY_NAME,
                )
                wandb.log({
                    'param_importances': param_importances,
                    'optimization_history': optimization_history,
                })
                wandb.finish()
            except:
                pass
            
            # Write output to results file
            best_params = (json.dumps(study.best_params, indent=12)
                           .replace('{\n', '')
                           .replace('\n}', ''))
            output = f'''
    Variable: price movement (up or down), model: {model.__name__}
        Best hyperparameters:
{best_params}
        Best value: {study.best_value:.2f} % excess return'''
        
            with open('optuna_results.txt', 'a') as f:
                f.write('\n' + output)

        for timeframe in (7, 14, 21):

            if timeframe == 7:
                iter_config = {
                    'name': 'n_estimators',
                    'low': 290,
                    'high': 320,
                }
                split_config = {
                    'name': 'min_samples_split',
                    'low': 9,
                    'high': 25,
                }
                leaf_config = {
                    'name': 'min_samples_leaf',
                    'low': 8,
                    'high': 11,
                }
                weight_config = {
                    'name': 'minority_class_weight',
                    'low': 5.1,
                    'high': 5.7,
                }
            else:
                continue
            
            # select only the variables with a statistically significant causal relationship
            with open(f'../../../4_eda/{coin}_causality/{coin}_price_min_{timeframe}d_causality.txt') as f:
                price_min_vars = f.read().splitlines()
            with open(f'../../../4_eda/{coin}_causality/{coin}_price_max_{timeframe}d_causality.txt') as f:
                price_max_vars = f.read().splitlines()
            X_min = data[price_min_vars]
            X_max = data[price_max_vars]

            # select targets
            min_target = f'{coin}_price_min_{timeframe}d'
            max_target = f'{coin}_price_max_{timeframe}d'
            y_min = targets[min_target]
            y_max = targets[max_target]

            # specify model, hyperparameter range for tuning and configuration
            # for cross validation
            model = RandomForestClassifier
                
            cv_config = {
                'add_constant_sliding_window': False,
                'cutoff_tuning': True,
                'tuning_method': 'accuracy',
            }

            print(f'\nRunning optimisation for {coin} {timeframe} days extrema...\n')

            # Define the objective function for optuna
            STUDY_NAME = f'{coin}_extrema_{timeframe}d_classification'
            def objective(trial):
                
                minority_class_weight = trial.suggest_float(**weight_config)
                crit = 'gini'
                    
                model_config = {
                    'criterion': crit,
                    'n_estimators': trial.suggest_int(**iter_config),
                    'min_samples_split': trial.suggest_int(**split_config),
                    'min_samples_leaf': trial.suggest_int(**leaf_config),
                    'class_weight': {0: 1, 1: minority_class_weight},
                }
                classifier_obj = model(**model_config, **base_config)

                config = {
                    'trial_number': trial.number,
                    'minority_class_weight': minority_class_weight,
                    **model_config,
                    **cv_config,
                }
                wandb.init(
                    project='Crypto RF',
                    config=config,
                    group=STUDY_NAME,
                    reinit=True,
                )
                result = crossvalidate_extrema(
                    classifier_obj,
                    X_min,
                    X_max,
                    y_min,
                    y_max,
                    price_data,
                    args.folds,
                    wandb_log=True,
                    **cv_config,
                )
                wandb.log(data={'avg_profit': result[3]})
                wandb.finish()
                return result[3]

            # Run optuna study
            study = optuna.create_study(
                direction='maximize',
                study_name=STUDY_NAME)
            study.optimize(objective, n_trials=args.iter)
            
            # Log optuna summary plots to W&B
            try:
                param_importances = optuna.visualization.plot_param_importances(study)
                optimization_history = optuna.visualization.plot_optimization_history(study)
                wandb.init(
                    project='Crypto RF',
                    group=STUDY_NAME,
                )
                wandb.log({
                    'param_importances': param_importances,
                    'optimization_history': optimization_history,
                })
                wandb.finish()
            except:
                pass
            
            # Write output to results file
            best_params = (json.dumps(study.best_params, indent=12)
                           .replace('{\n', '')
                           .replace('\n}', ''))
            output = f'''
    Variable: extrema (min/max), +/- {timeframe} days, model: {model.__name__}
        Best hyperparameters:
{best_params}
        Best value: {study.best_value:.2f} % excess return'''

            with open('optuna_results.txt', 'a') as f:
                f.write('\n' + output)

        with open('optuna_results.txt', 'a') as f:
            f.write('\n'*2)

    print('Done! Hyperparameter optimisation results saved as `optuna_results.txt`.')
    

if __name__=='__main__':
    
    hyperparameter_opt_rf()
