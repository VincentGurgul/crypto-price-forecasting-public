''' Functions for tuning Granger causality confidence level. '''

import sys
sys.path += ['../../../4_eda/', '../../']

import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

from granger_causality_functions import individual_granger_causality
from functions import *


def get_causal_vars(conf_level: float):

    btc_lagged = pd.read_parquet('../../../4_eda/btc_stationary_data_lagged.parquet.gzip')
    btc_lagged.columns = btc_lagged.columns.map('_'.join)
    btc_lagged = btc_lagged.fillna(method='ffill').fillna(0).replace([np.inf, -np.inf], 0)
    btc_targets = pd.read_parquet('../../../2_data_processing/numeric_data/btc_targets.parquet.gzip')

    directory = './btc_causality/'
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    for target in btc_targets.columns:
        data = pd.concat([btc_lagged, btc_targets[target]], axis=1)
        causality = individual_granger_causality(data, target, conf=conf_level)
        with open(directory + target + '_causality.txt', 'w') as f:
            f.write('\n'.join(np.sort(causality.variable.values)))
            
    eth_lagged = pd.read_parquet('../../../4_eda/eth_stationary_data_lagged.parquet.gzip')
    eth_lagged.columns = eth_lagged.columns.map('_'.join)
    eth_lagged = eth_lagged.fillna(method='ffill').fillna(0).replace([np.inf, -np.inf], 0)
    eth_targets = pd.read_parquet('../../../2_data_processing/numeric_data/eth_targets.parquet.gzip')

    directory = './eth_causality/'
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    for target in eth_targets.columns:
        data = pd.concat([eth_lagged, eth_targets[target]], axis=1)
        causality = individual_granger_causality(data, target, conf=conf_level)
        with open(directory + target + '_causality.txt', 'w') as f:
            f.write('\n'.join(np.sort(causality.variable.values)))
            
            
@ignore_warnings(category=ConvergenceWarning)
def run_all_models(args, base_config, CUTOFF_TUNING, TUNING_METHOD):
    
    all_target_results = []
    
    for coin in ('btc', 'eth'):
        
        price_data = pd.read_parquet(f'../../../2_data_processing/numeric_data/{coin}_price_data.parquet.gzip').iloc[:,0]
        
        for problem in ('regression', 'classification'):

            data_path = f'../../../4_eda/{coin}_stationary_data_lagged.parquet.gzip'
                
            data = pd.read_parquet(data_path)
            data.columns = data.columns.map('_'.join)
            data = (data.fillna(method='ffill')
                        .fillna(0)
                        .replace([np.inf, -np.inf], 0))
            target_path = f'../../../2_data_processing/numeric_data/{coin}_targets.parquet.gzip'
            targets = pd.read_parquet(target_path)

            with open(f'./{coin}_causality/{coin}_price_log_difference_causality.txt') as f:
                price_log_difference_vars = f.read().splitlines()
            X = data[price_log_difference_vars]
            
            y = targets[f'{coin}_price_log_difference']
            
            # Set optimal model hyperparameters for each context
            if problem == 'regression':
                model = RandomForestRegressor
                cv_config = {
                    'add_constant_sliding_window': args.double_cv,
                }
                if coin == 'btc':
                    model_config = {
                        'criterion': 'friedman_mse',
                        'n_estimators': 584,
                        'min_samples_split': 20,
                        'min_samples_leaf': 8,
                    }
                elif coin == 'eth':
                    model_config = {
                        'criterion': 'friedman_mse',
                        'n_estimators': 343,
                        'min_samples_split': 19,
                        'min_samples_leaf': 4,
                    }
            elif problem == 'classification':
                model = RandomForestClassifier
                cv_config = {
                    'add_constant_sliding_window': args.double_cv,
                    'cutoff_tuning': CUTOFF_TUNING,
                    'tuning_method': TUNING_METHOD,
                }
                if coin == 'btc':
                    model_config = {
                        'criterion': 'gini',
                        'n_estimators': 131,
                        'min_samples_split': 15,
                        'min_samples_leaf': 6,
                        'class_weight': {0: 1, 1: 3.682639189804975},
                    }
                elif coin == 'eth':
                    model_config = {
                        'criterion': 'gini',
                        'n_estimators': 634,
                        'min_samples_split': 8,
                        'min_samples_leaf': 1,
                        'class_weight': {0: 1, 1: 1.340297455745582},
                    }
                    
            print(f'Running crossvalidation for {coin} price movement {problem}...')
            result = crossvalidate_movement(
                model(**model_config, **base_config),
                problem,
                X,
                y,
                price_data,
                args.folds,
                **cv_config
            )
            
            all_target_results.append(result[3])

        for timeframe in (7, 14, 21):

            with open(f'../../../4_eda/{coin}_causality/{coin}_price_min_{timeframe}d_causality.txt') as f:
                price_min_vars = f.read().splitlines()
            with open(f'../../../4_eda/{coin}_causality/{coin}_price_max_{timeframe}d_causality.txt') as f:
                price_max_vars = f.read().splitlines()
            X_min = data[price_min_vars]
            X_max = data[price_max_vars]

            # select targets
            min_target = f'{coin}_price_min_{timeframe}d'
            max_target = f'{coin}_price_max_{timeframe}d'
            y_min = targets[min_target].fillna(0)
            y_max = targets[max_target].fillna(0)

            model = RandomForestClassifier
            cv_config = {
                'add_constant_sliding_window': args.double_cv,
                'cutoff_tuning': CUTOFF_TUNING,
                'tuning_method': TUNING_METHOD,
            }
            
            # Set optimal model hyperparameters for each context
            if coin == 'btc':
                if timeframe == 7:
                    model_config = {
                        'criterion': 'gini',
                        'n_estimators': 297,
                        'min_samples_split': 10,
                        'min_samples_leaf': 8,
                        'class_weight': {0: 1, 1: 5.603292259200591},
                    }
                if timeframe == 14:
                    model_config = {
                        'criterion': 'entropy',
                        'n_estimators': 468,
                        'min_samples_split': 8,
                        'min_samples_leaf': 10,
                        'class_weight': {0: 1, 1: 7.837826527004597},
                    }
                if timeframe == 21:
                    model_config = {
                        'criterion': 'gini',
                        'n_estimators': 722,
                        'min_samples_split': 13,
                        'min_samples_leaf': 9,
                        'class_weight': {0: 1, 1: 10.011914447080045},
                    }
            if coin == 'eth':
                if timeframe == 7:
                    model_config = {
                        'criterion': 'gini',
                        'n_estimators': 108,
                        'min_samples_split': 19,
                        'min_samples_leaf': 1,
                        'class_weight': {0: 1, 1: 2.070922239254963},
                    }
                if timeframe == 14:
                    model_config = {
                        'criterion': 'gini',
                        'n_estimators': 260,
                        'min_samples_split': 8,
                        'min_samples_leaf': 1,
                        'class_weight': {0: 1, 1: 7.879301742843994},
                    }
                if timeframe == 21:
                    model_config = {
                        'criterion': 'gini',
                        'n_estimators': 479,
                        'min_samples_split': 12,
                        'min_samples_leaf': 4,
                        'class_weight': {0: 1, 1: 3.7891243592149717},
                    }

            print(f'Running crossvalidation for {coin} {timeframe} days extrema...')
            result = crossvalidate_extrema(
                model(**model_config, **base_config),
                X_min,
                X_max,
                y_min,
                y_max,
                price_data,
                args.folds,
                **cv_config,
            )
            
            all_target_results.append(result[3])
            
    return all_target_results