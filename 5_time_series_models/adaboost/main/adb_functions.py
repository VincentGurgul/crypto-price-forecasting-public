''' Functions for forecasting crypto prices with Adaboost with optimal
hyperparameters. '''

import sys
sys.path += ['../../']

import re
import numpy as np
import pandas as pd

import argparse
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier

from functions import *


def run_and_log_adaboost(mode: str,
                         coin: str,
                         args: argparse.Namespace,
                         CUTOFF_TUNING: bool = True,
                         TUNING_METHOD: str = 'accuracy'):
    
    results_file = f'results_{mode}.txt'
    base_config = {
        'random_state': 42,
    }
    
    price_data = pd.read_parquet(f'../../../2_data_processing/numeric_data/{coin}_price_data.parquet.gzip').iloc[:,0]
    
    with open(results_file, 'a') as f:
        f.write(f'----------------\nCoin: {coin.upper()}\n----------------')

    if mode in ['full_untransformed', 'causal_nonstationary']:
        data_path = f'../../../4_eda/{coin}_data_lagged.parquet.gzip'
    else:
        data_path = f'../../../4_eda/{coin}_stationary_data_lagged.parquet.gzip'
        
    data = pd.read_parquet(data_path)
    data.columns = data.columns.map('_'.join)
    data = (data.fillna(method='ffill')
                .fillna(0)
                .replace([np.inf, -np.inf], 0))
    target_path = f'../../../2_data_processing/numeric_data/{coin}_targets.parquet.gzip'
    targets = pd.read_parquet(target_path)
    
    with open(f'../../../4_eda/{coin}_causality/{coin}_price_log_difference_causality.txt') as f:
        price_log_difference_vars = f.read().splitlines()
        
    if mode == 'causal_nonstationary':
        l = []
        for i in price_log_difference_vars:
            updated_var_name = re.sub(r'_d\d*_', '_', i)
            l.append(updated_var_name)
        price_log_difference_vars = l
        
    # select only the desired explanatory variables
    if mode in ['causal_stationary_no_nlp', 'causal_nonstationary']:
        X = data[price_log_difference_vars]
        columns_to_keep = [
            col for col in X.columns
            if 'bart_mnli' not in col and 'roberta' not in col
        ]
        X = X[columns_to_keep]
    elif mode == 'causal_stationary_nlp_pretrained':
        X = data[price_log_difference_vars]
        columns_to_keep = [
            col for col in X.columns if 'finetuned' not in col
        ]
        X = X[columns_to_keep]
    elif mode == 'causal_stationary_nlp_finetuned':
        X = data[price_log_difference_vars]
        columns_to_keep = [
            col for col in X.columns
            if 'bart_mnli' not in col and 'roberta_pretrained' not in col
        ]
        X = X[columns_to_keep]
    else:
        columns_to_keep = [
            col for col in data.columns
            if 'bart_mnli' not in col and 'roberta' not in col
        ]
        X = data[columns_to_keep]
    
    # select the target variable
    y = targets[f'{coin}_price_log_difference']
        
    # Predict daily price movement as both regression and classification problem
    for problem in ('regression', 'classification'):
        
        # Set optimal model hyperparameters for each context
        if problem == 'regression':
            model = AdaBoostRegressor
            cv_config = {
                'add_constant_sliding_window': args.double_cv,
            }
            if coin == 'btc':
                model_config = {
                    'estimator': DecisionTreeRegressor(max_depth=5),
                    'n_estimators': 723,
                    'learning_rate': 1.2773564759107672,
                    'loss': 'square',
                }
            elif coin == 'eth':
                model_config = {
                    'estimator': DecisionTreeRegressor(max_depth=1),
                    'n_estimators': 730,
                    'learning_rate': 0.8941325712727377,
                    'loss': 'linear',
                }
        elif problem == 'classification':
            model = AdaBoostClassifier
            cv_config = {
                'add_constant_sliding_window': args.double_cv,
                'cutoff_tuning': CUTOFF_TUNING,
                'tuning_method': TUNING_METHOD,
            }
            if coin == 'btc':
                model_config = {
                    'estimator': DecisionTreeClassifier(max_depth=1),
                    'n_estimators': 300,
                    'learning_rate': 0.4089349325540496,
                    'algorithm': 'SAMME',
                }
            elif coin == 'eth':
                model_config = {
                    'estimator': DecisionTreeClassifier(max_depth=9),
                    'n_estimators': 642,
                    'learning_rate': 1.8616214490480256,
                    'algorithm': 'SAMME.R'
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

        output = f'''
    Variable: price movement (up or down), model: {model.__name__}
    Parameters:
        {model_config}
    Mean results:
        Baseline profit: {result[0]:.2f} % with 2 trades
        Prediction profit: {result[1]:.2f} % with ⌀ {result[2]:.1f} trades 
            Difference: {result[3]:.2f} %
        Target profit: {result[4]:.2f} % with ⌀ {result[5]:.1f} trades 
        Prediction AUC: {result[6]:.4f}, accuracy: {result[7]:.4f}'''
    
        try:
            output += f' | Cutoff: {result[8]:.3f}'
        except:
            pass

        with open(results_file, 'a') as f:
            f.write('\n' + output)

    # Classify local extremepoints for three different underlying timeframes
    for timeframe in (7, 14, 21):

        with open(f'../../../4_eda/{coin}_causality/{coin}_price_min_{timeframe}d_causality.txt') as f:
            price_min_vars = f.read().splitlines()
        with open(f'../../../4_eda/{coin}_causality/{coin}_price_max_{timeframe}d_causality.txt') as f:
            price_max_vars = f.read().splitlines()
            
        if mode == 'causal_nonstationary':
            l = []
            for i in price_min_vars:
                updated_var_name = re.sub(r'_d\d*_', '_', i)
                l.append(updated_var_name)
            price_min_vars = l
            l = []
            for i in price_max_vars:
                updated_var_name = re.sub(r'_d\d*_', '_', i)
                l.append(updated_var_name)
            price_max_vars = l
            
        # select only the desired explanatory variables
        if mode in ['causal_stationary_no_nlp', 'causal_nonstationary']:
            X_min = data[price_min_vars]
            X_max = data[price_max_vars]
            columns_to_keep = [
                col for col in X_min.columns
                if 'bart_mnli' not in col and 'roberta' not in col
            ]
            X_min = X_min[columns_to_keep]
            columns_to_keep = [
                col for col in X_max.columns
                if 'bart_mnli' not in col and 'roberta' not in col
            ]
            X_max = X_max[columns_to_keep]
        elif mode == 'causal_stationary_nlp_pretrained':
            X_min = data[price_min_vars]
            X_max = data[price_max_vars]
            columns_to_keep = [
                col for col in X_min.columns if 'finetuned' not in col
            ]
            X_min = X_min[columns_to_keep]
            columns_to_keep = [
                col for col in X_max.columns if 'finetuned' not in col
            ]
            X_max = X_max[columns_to_keep]
        elif mode == 'causal_stationary_nlp_finetuned':
            X_min = data[price_min_vars]
            X_max = data[price_max_vars]
            columns_to_keep = [
                col for col in X_min.columns
                if 'bart_mnli' not in col and 'roberta_pretrained' not in col
            ]
            X_min = X_min[columns_to_keep]
            columns_to_keep = [
                col for col in X_max.columns
                if 'bart_mnli' not in col and 'roberta_pretrained' not in col
            ]
            X_max = X_max[columns_to_keep]
        else:
            columns_to_keep = [
                col for col in data.columns
                if 'bart_mnli' not in col and 'roberta' not in col
            ]
            X_min = data[columns_to_keep]
            X_max = data[columns_to_keep]

        # select target variables
        min_target = f'{coin}_price_min_{timeframe}d'
        max_target = f'{coin}_price_max_{timeframe}d'
        y_min = targets[min_target]
        y_max = targets[max_target]

        model = AdaBoostClassifier
        cv_config = {
            'add_constant_sliding_window': args.double_cv,
            'cutoff_tuning': CUTOFF_TUNING,
            'tuning_method': TUNING_METHOD,
        }
        
        # Set optimal model hyperparameters for each context
        if coin == 'btc':
            if timeframe == 7:
                model_config = {
                    'estimator': DecisionTreeClassifier(max_depth=1),
                    'n_estimators': 438,
                    'learning_rate': 1.8377494349947938,
                    'algorithm': 'SAMME.R'
                }
            if timeframe == 14:
                model_config = {
                    'estimator': DecisionTreeClassifier(max_depth=1),
                    'n_estimators': 478,
                    'learning_rate': 0.6911635622904206,
                    'algorithm': 'SAMME.R'
                }
            if timeframe == 21:
                model_config = {
                    'estimator': DecisionTreeClassifier(max_depth=2),
                    'n_estimators': 124,
                    'learning_rate': 0.5636758688349135,
                    'algorithm': 'SAMME',
                }
        if coin == 'eth':
            if timeframe == 7:
                model_config = {
                    'estimator': DecisionTreeClassifier(max_depth=2),
                    'n_estimators': 356,
                    'learning_rate': 1.7291778122826171,
                    'algorithm': 'SAMME',
                }
            if timeframe == 14:
                model_config = {
                    'estimator': DecisionTreeClassifier(max_depth=2),
                    'n_estimators': 525,
                    'learning_rate': 0.5264463238920566,
                    'algorithm': 'SAMME.R',
                }
            if timeframe == 21:
                model_config = {
                    'estimator': DecisionTreeClassifier(max_depth=3),
                    'n_estimators': 101,
                    'learning_rate': 0.7506674720106858,
                    'algorithm': 'SAMME',
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
        
        output = f'''
    Variable: extrema (min/max), +/- {timeframe} days, model: {model.__name__}
    Parameters:
        {model_config}
    Mean results:
        Baseline profit: {result[0]:.2f} % with 2 trades
        Prediction profit: {result[1]:.2f} % with ⌀ {result[2]:.1f} trades 
            Difference: {result[3]:.2f} %
        Target profit: {result[4]:.2f} % with ⌀ {result[5]:.1f} trades 
        Minima AUC: {result[6]:.4f} | Maxima AUC: {result[7]:.4f} | Mean: {np.mean([result[6], result[7]]):.4f}
        Minima accuracy: {result[8]:.4f} | Maxima accuracy: {result[9]:.4f} | Mean: {np.mean([result[8], result[9]]):.4f}
        '''
        try:
            output += f'Minima cutoff: {result[10]:.3f} | Maxima cutoff: {result[11]:.3f}'
        except:
            pass
    
        with open(results_file, 'a') as f:
            f.write('\n' + output)
            
    with open(results_file, 'a') as f:
        f.write('\n'*2)
