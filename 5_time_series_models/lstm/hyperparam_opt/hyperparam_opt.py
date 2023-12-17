''' Hyperparameter optimisation for crypto price forecasting with LSTM. '''

import sys
sys.path += ['../', '../../', '../../../']

import numpy as np
import pandas as pd
import json
import optuna
import wandb

from argparse import ArgumentParser
from lstm_functions import (
    crossvalidate_movement_lstm,
    crossvalidate_extrema_lstm                     
)
from lag_functions import shift_timeseries_by_lags
from functions import *

from utils.wrappers import timeit, telegram_notify


@timeit
@telegram_notify
def hyperparameter_opt_lstm():
    
    modes = [
        'causal_stationary_vader',
        'causal_stationary_nlp_pretrained',
        'causal_stationary_twitter_roberta',
        'causal_stationary_bart_mnli',
        'causal_stationary_nlp_finetuned',
        'causal_stationary_no_nlp',
        'causal_stationary_full_data',
        'full_stationary',
    ]
    
    # parse cross validation folds and optuna iterations as args
    parser = ArgumentParser(
        description='Crypto price forecasting with LSTM'
    )
    parser.add_argument(
        '-m',
        '--mode',
        type=str,
        default='causal_stationary_nlp_pretrained',
        help=f'(str) mode for the analysis, available modes: {modes}, defaults to `causal_stationary_nlp_pretrained`',
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
        default=200,
        help='number of trial runs per optuna run, i.e. per model, default = 200',
    )
    args = parser.parse_args()

    print(f'\n{args}\n')
    
    if args.mode not in modes:
        raise ValueError(f'Selected mode `{args.mode}` is invalid. Please select one of the following modes: {modes}')

    # Print args to results file
    with open(f'optuna_results_{args.mode}.txt', 'w') as f:
        f.write(f'''Args:
    Optimising profit based on time series cross validation with {args.folds*2} folds.
    Optuna running {args.iter} trial runs per model.\n\n''')
    
    for coin in ('btc', 'eth'):
        
        cv_config = {
            'add_constant_sliding_window': False,
            'cutoff_tuning': False,
            'skip_first': True,
        }
        
        price_data = pd.read_parquet(f'../../../2_data_processing/numeric_data/{coin}_price_data.parquet.gzip').iloc[:,0]
        
        with open(f'optuna_results_{args.mode}.txt', 'a') as f:
            f.write(f'----------------\nCoin: {coin.upper()}\n----------------')
            
        numeric_data = pd.read_parquet(f'../../../2_data_processing/numeric_data/{coin}_numeric_stationary_data.parquet.gzip')
        nlp_data = pd.read_parquet(f'../../../3_nlp_models/4_processing/{coin}_stationary_text_data.parquet.gzip')
        targets = pd.read_parquet(f'../../../2_data_processing/numeric_data/{coin}_targets.parquet.gzip')

        if coin == 'btc':
            data = (pd.concat([numeric_data, nlp_data], axis=1)
                    .loc[1314662400:1678752000])
        elif coin == 'eth':
            data = (pd.concat([numeric_data, nlp_data], axis=1)
                    .loc[1445472000:1678838400])
        data = (data.fillna(method='ffill')
                    .fillna(0)
                    .replace([np.inf, -np.inf], 0))
        
        with open(f'../../../4_eda/{coin}_lstm_causality/{coin}_price_log_difference_causality.txt') as f:
            price_log_difference_vars = f.read().splitlines()
            
        # select only the desired explanatory variables
        if args.mode == 'causal_stationary_no_nlp':
            X = data[price_log_difference_vars]
            columns_to_keep = [
                col for col in X.columns
                if all(i not in col for i in ['bart_mnli', 'roberta', 'vader'])
            ]
            X = X[columns_to_keep]
        elif args.mode == 'causal_stationary_vader':
            X = data[price_log_difference_vars]
            columns_to_keep = [
                col for col in X.columns
                if all(i not in col for i in ['bart_mnli', 'roberta'])
            ]
            X = X[columns_to_keep]
        elif args.mode == 'causal_stationary_twitter_roberta':
            X = data[price_log_difference_vars]
            columns_to_keep = [
                col for col in X.columns
                if all(i not in col for i in ['finetuned', 'bart_mnli', 'vader'])
            ]
            X = X[columns_to_keep]
        elif args.mode == 'causal_stationary_bart_mnli':
            X = data[price_log_difference_vars]
            columns_to_keep = [
                col for col in X.columns
                if all(i not in col for i in ['roberta', 'vader'])
            ]
            X = X[columns_to_keep]
        elif args.mode == 'causal_stationary_nlp_pretrained':
            X = data[price_log_difference_vars]
            columns_to_keep = [
                col for col in X.columns
                if all(i not in col for i in ['finetuned', 'vader'])
            ]
            X = X[columns_to_keep]
        elif args.mode == 'causal_stationary_nlp_finetuned':
            X = data[price_log_difference_vars]
            columns_to_keep = [
                col for col in X.columns
                if all(i not in col for i in ['bart_mnli', 'roberta_pretrained', 'vader'])
            ]
            X = X[columns_to_keep]
        else:
            columns_to_keep = [
                col for col in data.columns
                if all(i not in col for i in ['bart_mnli', 'roberta', 'vader'])
            ]
            X = data[columns_to_keep]

        # Lag the features and select the target
        X = shift_timeseries_by_lags(X, np.arange(0, 14))
        y = targets[f'{coin}_price_log_difference']

        for problem in ('regression', 'classification'):
            
            # specify configuration for cross validation
            if problem == 'classification':
                cv_config['cutoff_tuning'] = True
                cv_config['tuning_method'] = 'accuracy'
            
            print(f'\nRunning optimisation for {coin} price movement {problem}...\n')
            
            # Define the objective function for optuna
            STUDY_NAME = f'{coin}_movement_{problem}'
            def objective(trial):
                
                lstm_layer_sizes = []
                for i in range(trial.suggest_int('lstm_layers', 1, 3)):
                    lstm_layer_sizes.append(
                        trial.suggest_int(f'lstm_units_{i}', 50, 300)
                    )
                dense_layer_sizes = []
                for i in range(trial.suggest_int('dense_layers', 0, 3)):
                    dense_layer_sizes.append(
                        trial.suggest_int(f'dense_units_{i}', 10, 150)
                    )
                    
                model_args = {
                    'lstm_layer_sizes': lstm_layer_sizes,
                    'dense_layer_sizes': dense_layer_sizes,
                    'activation': trial.suggest_categorical(
                        'activation', ['relu', 'tanh'],
                    ),
                    'dropout': trial.suggest_float(
                        'dropout', 0.1, 0.5,
                    ),
                    'optimizer': trial.suggest_categorical(
                        'optimizer', ['Adam', 'RMSprop', 'SGD'],
                    ),
                    'learning_rate': trial.suggest_float(
                        'learning_rate', 1e-4, 1e-1, log=True,
                    ),
                    'scaling': trial.suggest_categorical(
                        'scaling', [None, 'StandardScaler', 'MinMaxScaler'],
                    ),
                }
                training_args = {
                    'epochs': trial.suggest_int(
                        'epochs', 10, 200,
                    ),
                    'batch_size': trial.suggest_categorical(
                        'batch_size', [32, 64, 128, 256],
                    ),
                    'verbosity': 0,
                }
                if problem == 'classification':
                    model_args['binary_classification'] = True
                    
                log_config = {
                    'trial_number': trial.number,
                    'lstm_layers': len(lstm_layer_sizes),
                    'dense_layers': len(dense_layer_sizes),
                    **{f'lstm_units_{i}': n_units for i, n_units in enumerate(lstm_layer_sizes)},
                    **{f'dense_units_{i}': n_units for i, n_units in enumerate(dense_layer_sizes)},
                    **model_args,
                    **training_args,
                    **cv_config,
                }
                wandb.init(
                    project=f'Crypto LSTM {args.mode}',
                    config=log_config,
                    group=STUDY_NAME,
                    reinit=True,
                )
                result = crossvalidate_movement_lstm(
                    model_args,
                    training_args,
                    problem,
                    X,
                    y,
                    price_data,
                    args.folds,
                    wandb_log=True,
                    **cv_config,
                )
                wandb.log(data={'avg_trades': result[2]})
                wandb.log(data={'avg_profit': result[3]})
                wandb.log(data={'avg_auc_roc': result[6]})
                wandb.log(data={'avg_accuracy': result[7]})
                try:
                    wandb.log(data={'avg_cutoff': result[8]})
                except:
                    pass
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
                    project=f'Crypto LSTM {args.mode}',
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
    Variable: price movement (up or down), model: LSTM
        Best hyperparameters:
{best_params}
        Best value: {study.best_value:.2f} % excess return'''
        
            with open(f'optuna_results_{args.mode}.txt', 'a') as f:
                f.write('\n' + output)

        for timeframe in (7, 14, 21):
            
            # Set configuration for cross validation
            cv_config['cutoff_tuning'] = True
            cv_config['tuning_method'] = 'accuracy'

            with open(f'../../../4_eda/{coin}_lstm_causality/{coin}_price_min_{timeframe}d_causality.txt') as f:
                price_min_vars = f.read().splitlines()
            with open(f'../../../4_eda/{coin}_lstm_causality/{coin}_price_max_{timeframe}d_causality.txt') as f:
                price_max_vars = f.read().splitlines()
                
            # select only the desired explanatory variables
            if args.mode == 'causal_stationary_no_nlp':
                X_min = data[price_min_vars]
                X_max = data[price_max_vars]
                columns_to_keep = [
                    col for col in X_min.columns
                    if all(i not in col for i in ['bart_mnli', 'roberta', 'vader'])
                ]
                X_min = X_min[columns_to_keep]
                columns_to_keep = [
                    col for col in X_max.columns
                    if all(i not in col for i in ['bart_mnli', 'roberta', 'vader'])
                ]
                X_max = X_max[columns_to_keep]
            elif args.mode == 'causal_stationary_vader':
                X_min = data[price_min_vars]
                X_max = data[price_max_vars]
                columns_to_keep = [
                    col for col in X_min.columns
                    if all(i not in col for i in ['bart_mnli', 'roberta'])
                ]
                X_min = X_min[columns_to_keep]
                columns_to_keep = [
                    col for col in X_max.columns
                    if all(i not in col for i in ['bart_mnli', 'roberta'])
                ]
                X_max = X_max[columns_to_keep]
            elif args.mode == 'causal_stationary_twitter_roberta':
                X_min = data[price_min_vars]
                X_max = data[price_max_vars]
                columns_to_keep = [
                    col for col in X_min.columns
                    if all(i not in col for i in ['finetuned', 'bart_mnli', 'vader'])
                ]
                X_min = X_min[columns_to_keep]
                columns_to_keep = [
                    col for col in X_max.columns
                    if all(i not in col for i in ['finetuned', 'bart_mnli', 'vader'])
                ]
                X_max = X_max[columns_to_keep]
            elif args.mode == 'causal_stationary_bart_mnli':
                X_min = data[price_min_vars]
                X_max = data[price_max_vars]
                columns_to_keep = [
                    col for col in X_max.columns
                    if all(i not in col for i in ['roberta', 'vader'])
                ]
                X_min = X_min[columns_to_keep]
                columns_to_keep = [
                    col for col in X_max.columns
                    if all(i not in col for i in ['roberta', 'vader'])
                ]
                X_max = X_max[columns_to_keep]
            elif args.mode == 'causal_stationary_nlp_pretrained':
                X_min = data[price_min_vars]
                X_max = data[price_max_vars]
                columns_to_keep = [
                    col for col in X_min.columns
                    if all(i not in col for i in ['finetuned', 'vader'])
                ]
                X_min = X_min[columns_to_keep]
                columns_to_keep = [
                    col for col in X_min.columns
                    if all(i not in col for i in ['finetuned', 'vader'])
                ]
                X_max = X_max[columns_to_keep]
            elif args.mode == 'causal_stationary_nlp_finetuned':
                X_min = data[price_min_vars]
                X_max = data[price_max_vars]
                columns_to_keep = [
                    col for col in X_min.columns
                    if all(i not in col for i in ['bart_mnli', 'roberta_pretrained', 'vader'])
                ]
                X_min = X_min[columns_to_keep]
                columns_to_keep = [
                    col for col in X_max.columns
                    if all(i not in col for i in ['bart_mnli', 'roberta_pretrained', 'vader'])
                ]
                X_max = X_max[columns_to_keep]
            else:
                columns_to_keep = [
                    col for col in data.columns
                    if all(i not in col for i in ['bart_mnli', 'roberta', 'vader'])
                ]
                X_min = data[columns_to_keep]
                X_max = data[columns_to_keep]

            # Lag the features
            X_min = shift_timeseries_by_lags(X_min, np.arange(0, 14))
            X_max = shift_timeseries_by_lags(X_max, np.arange(0, 14))

            # Select targets
            min_target = f'{coin}_price_min_{timeframe}d'
            max_target = f'{coin}_price_max_{timeframe}d'
            y_min = targets[min_target]
            y_max = targets[max_target]

            print(f'\nRunning optimisation for {coin} {timeframe} days extrema...\n')

            # Define the objective function for optuna
            STUDY_NAME = f'{coin}_extrema_{timeframe}d_classification'
            def objective(trial):
                
                lstm_layer_sizes = []
                for i in range(trial.suggest_int('lstm_layers', 1, 3)):
                    lstm_layer_sizes.append(
                        trial.suggest_int(f'lstm_units_{i}', 50, 300)
                    )
                dense_layer_sizes = []
                for i in range(trial.suggest_int('dense_layers', 0, 3)):
                    dense_layer_sizes.append(
                        trial.suggest_int(f'dense_units_{i}', 10, 150)
                    )
                    
                model_args = {
                    'lstm_layer_sizes': lstm_layer_sizes,
                    'dense_layer_sizes': dense_layer_sizes,
                    'activation': trial.suggest_categorical(
                        'activation', ['relu', 'tanh'],
                    ),
                    'dropout': trial.suggest_float(
                        'dropout', 0.1, 0.5,
                    ),
                    'optimizer': trial.suggest_categorical(
                        'optimizer', ['Adam', 'RMSprop', 'SGD'],
                    ),
                    'learning_rate': trial.suggest_float(
                        'learning_rate', 1e-4, 1e-1, log=True,
                    ),
                    'scaling': trial.suggest_categorical(
                        'scaling', [None, 'StandardScaler', 'MinMaxScaler'],
                    ),
                    'binary_classification': True,
                }
                training_args = {
                    'minority_class_weight': trial.suggest_float(
                        'minority_class_weight', 1., 20.,
                    ),
                    'epochs': trial.suggest_int(
                        'epochs', 10, 200,
                    ),
                    'batch_size': trial.suggest_categorical(
                        'batch_size', [32, 64, 128, 256],
                    ),
                    'verbosity': 0,
                }
                
                log_config = {
                    'trial_number': trial.number,
                    'lstm_layers': len(lstm_layer_sizes),
                    'dense_layers': len(dense_layer_sizes),
                    **{f'lstm_units_{i}': n_units for i, n_units in enumerate(lstm_layer_sizes)},
                    **{f'dense_units_{i}': n_units for i, n_units in enumerate(dense_layer_sizes)},
                    **model_args,
                    **training_args,
                    **cv_config,
                }
                wandb.init(
                    project=f'Crypto LSTM {args.mode}',
                    config=log_config,
                    group=STUDY_NAME,
                    reinit=True,
                )
                result = crossvalidate_extrema_lstm(
                    model_args,
                    training_args,
                    X_min,
                    X_max,
                    y_min,
                    y_max,
                    price_data,
                    args.folds,
                    wandb_log=True,
                    **cv_config,
                )
                wandb.log(data={'avg_trades': result[2]})
                wandb.log(data={'avg_profit': result[3]})
                wandb.log(data={'avg_auc_roc':
                    np.mean([result[6], result[7]])})
                wandb.log(data={'avg_accuracy':
                    np.mean([result[8], result[9]])})
                try:
                    wandb.log(data={'avg_cutoff':
                        np.mean([result[10], result[11]])})
                except:
                    pass
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
                    project=f'Crypto LSTM {args.mode}',
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
    Variable: extrema (min/max), +/- {timeframe} days, model: LSTM

        Best hyperparameters:
{best_params}
        Best value: {study.best_value:.2f} % excess return'''

            with open(f'optuna_results_{args.mode}.txt', 'a') as f:
                f.write('\n' + output)

        with open(f'optuna_results_{args.mode}.txt', 'a') as f:
            f.write('\n'*2)

    print(f'Done! Hyperparameter optimisation results saved as `optuna_results_{args.mode}.txt`.')
    

if __name__=='__main__':
    
    hyperparameter_opt_lstm()
