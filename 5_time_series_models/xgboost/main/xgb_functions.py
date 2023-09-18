''' Functions for forecasting crypto prices with XGBoost with optimal
hyperparameters. '''

import sys
sys.path += ['../../']

import os
import numpy as np
import pandas as pd

import argparse
from xgboost import XGBRegressor, XGBClassifier

from functions import *
from config import get_model_config


def run_and_log_xgb(mode: str,
                    coin: str,
                    args: argparse.Namespace,
                    CUTOFF_TUNING: bool = True,
                    TUNING_METHOD: str = 'accuracy'):
    
    results_file = f'results_{mode}.txt'
    cv_config = {
        'add_constant_sliding_window': args.double_cv,
        'constant_window_size': 900 if coin == 'btc' else 600,
        'cutoff_tuning': False,
        'skip_first': True,
    }
    
    price_data = pd.read_parquet(f'../../../2_data_processing/numeric_data/{coin}_price_data.parquet.gzip').iloc[:,0]
    
    results_df = pd.DataFrame(columns=('profit', 'trades', 'auc roc', 'accuracy'))
    results_df.loc[coin] = np.nan
    
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
    
    if mode == 'causal_nonstationary':
        with open(f'../../../4_eda/{coin}_causality_nonstat/{coin}_price_log_difference_causality.txt') as f:
            price_log_difference_vars = f.read().splitlines()
    else:
        with open(f'../../../4_eda/{coin}_causality/{coin}_price_log_difference_causality.txt') as f:
            price_log_difference_vars = f.read().splitlines()
        
    # select only the desired explanatory variables
    if mode in ['causal_stationary_no_nlp', 'causal_nonstationary']:
        X = data[price_log_difference_vars]
        columns_to_keep = [
            col for col in X.columns
            if 'bart_mnli' not in col and 'roberta' not in col
        ]
        X = X[columns_to_keep]
    elif mode == 'causal_stationary_twitter_roberta':
        X = data[price_log_difference_vars]
        columns_to_keep = [
            col for col in X.columns
            if 'finetuned' not in col and 'bart_mnli' not in col
        ]
        X = X[columns_to_keep]
    elif mode == 'causal_stationary_bart_mnli':
        X = data[price_log_difference_vars]
        columns_to_keep = [
            col for col in X.columns if 'roberta' not in col
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
            model = XGBRegressor
        elif problem == 'classification':
            model = XGBClassifier
            cv_config['cutoff_tuning'] = CUTOFF_TUNING
            cv_config['tuning_method'] = TUNING_METHOD
                
        model_config = get_model_config(coin, problem)
                
        print(f'Running crossvalidation for {coin} price movement {problem}...')
        result = crossvalidate_movement(
            model(**model_config),
            problem,
            X,
            y,
            price_data,
            args.folds,
            **cv_config
        )

        variable = f'price movement (up or down), model: {model.__name__}'

        output = f'''
    Variable: {variable}
    Parameters:
        {model_config}
    Mean results:
        Baseline profit: {result[0]:.2f} % with 2 trades
        Prediction profit: {result[1]:.2f} % with ⌀ {result[2]:.1f} trades 
            Difference: {result[3]:.2f} %
        Target profit: {result[4]:.2f} % with ⌀ {result[5]:.1f} trades 
        Prediction AUC: {result[6]:.4f}, accuracy: {result[7]:.4f}'''
    
        results_df.loc[variable] = (str(result[3]) + '%',
                                    result[2], result[6], result[7])
        
        try:
            output += f' | Cutoff: {result[8]:.3f}'
        except:
            pass

        with open(results_file, 'a') as f:
            f.write('\n' + output)

    # Classify local extremepoints for three different underlying timeframes
    for timeframe in (7, 14, 21):

        if mode == 'causal_nonstationary':
            with open(f'../../../4_eda/{coin}_causality_nonstat/{coin}_price_min_{timeframe}d_causality.txt') as f:
                price_min_vars = f.read().splitlines()
            with open(f'../../../4_eda/{coin}_causality_nonstat/{coin}_price_max_{timeframe}d_causality.txt') as f:
                price_max_vars = f.read().splitlines()
        else:
            with open(f'../../../4_eda/{coin}_causality/{coin}_price_min_{timeframe}d_causality.txt') as f:
                price_min_vars = f.read().splitlines()
            with open(f'../../../4_eda/{coin}_causality/{coin}_price_max_{timeframe}d_causality.txt') as f:
                price_max_vars = f.read().splitlines()
            
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
        elif mode == 'causal_stationary_twitter_roberta':
            X_min = data[price_min_vars]
            X_max = data[price_max_vars]
            columns_to_keep = [
                col for col in X_min.columns
                if 'finetuned' not in col and 'bart_mnli' not in col
            ]
            X_min = X_min[columns_to_keep]
            columns_to_keep = [
                col for col in X_max.columns
                if 'finetuned' not in col and 'bart_mnli' not in col
            ]
            X_max = X_max[columns_to_keep]
        elif mode == 'causal_stationary_bart_mnli':
            X_min = data[price_min_vars]
            X_max = data[price_max_vars]
            columns_to_keep = [
                col for col in X_min.columns if 'roberta' not in col
            ]
            X_min = X_min[columns_to_keep]
            columns_to_keep = [
                col for col in X_max.columns if 'roberta' not in col
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

        # define model and hyperparamter configuration
        model = XGBClassifier
        model_config = get_model_config(coin, 'extrema', timeframe)

        print(f'Running crossvalidation for {coin} {timeframe} days extrema...')
        result = crossvalidate_extrema(
            model(**model_config),
            X_min,
            X_max,
            y_min,
            y_max,
            price_data,
            args.folds,
            **cv_config,
        )
        
        variable = f'extrema (min/max), +/- {timeframe} days, model: {model.__name__}'
        
        output = f'''
    Variable: {variable}
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
        
        results_df.loc[variable] = (str(result[3]) + '%',
                                    result[2],
                                    np.mean([result[6], result[7]]),
                                    np.mean([result[8], result[9]]))
        
        try:
            output += f'Minima cutoff: {result[10]:.3f} | Maxima cutoff: {result[11]:.3f}'
        except:
            pass
    
        with open(results_file, 'a') as f:
            f.write('\n' + output)
            
    with open(results_file, 'a') as f:
        f.write('\n'*2)
        
    # Save results dataframe
    if not os.path.exists('results_csv'):
        os.makedirs('results_csv')
    results_df.to_csv(f'results_csv/{mode}_{coin}.csv')
