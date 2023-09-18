''' Hyperparameter optimisation for crypto price forecasting of Temporal
Fusion Transformer with GPU utilisation.

Recommended running only on one GPU with
>>> CUDA_VISIBLE_DEVICES=0 python hyperparam_opt.py --gpus 1

Note: Because there has been a change in how pytorch lightning is imported one
needs to make a change in the pytorch_forecasting source code to make this
script work:
- Go to the folder of your python version, e.g. /python3.8/
- Open the file /site-packages/pytorch_forecasting/models/base_models.py
- Replace all imports `lightning.pytorch` with `pytorch_lightning`
'''

import sys
sys.path += ['../', '../../']

import numpy as np
import pandas as pd
import json
import optuna
import wandb

from argparse import ArgumentParser

from tft_functions import (
    crossvalidate_extrema_tft,
    crossvalidate_movement_tft,
)
from functions import *

from utils.wrappers import timeit, telegram_notify


@timeit
@telegram_notify
def hyperparameter_opt_tft():
    
    modes = [
        'stationary_nlp_pretrained',
        'stationary_twitter_roberta',
        'stationary_bart_mnli',
        'stationary_nlp_finetuned',
        'stationary_no_nlp',
        'stationary_full_data',
    ]
    
    # parse cross validation folds and optuna iterations as args
    parser = ArgumentParser(
        description='Crypto price forecasting with TFT'
    )
    parser.add_argument(
        '-g',
        '--gpus',
        type=int,
        default=1,
        help=f'(int) number of GPUs to use for calculation, either 1 or 2, defaults to 1',
    )
    parser.add_argument(
        '-m',
        '--mode',
        type=str,
        default='stationary_nlp_pretrained',
        help=f'(str) mode for the analysis, available modes: {modes}, defaults to `stationary_nlp_pretrained`',
    )
    parser.add_argument(
        '-f',
        '--folds',
        type=int,
        default=7,
        help='(int) number of folds for each time series cross validation, defaults to 7',
    )
    parser.add_argument(
        '-i',
        '--iter',
        type=int,
        default=200,
        help='(int) number of trial runs per optuna run, i.e. per model, defaults to 50',
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
        
        price_data = pd.read_parquet(f'../../2_data_processing/numeric_data/{coin}_price_data.parquet.gzip').iloc[:,0]
        
        with open(f'optuna_results_{args.mode}.txt', 'a') as f:
            f.write(f'----------------\nCoin: {coin.upper()}\n----------------')

        # fetch data
        numeric_data = pd.read_parquet(f'../../2_data_processing/numeric_data/{coin}_numeric_stationary_data.parquet.gzip')
        nlp_data = pd.read_parquet(f'../../3_nlp_models/4_processing/{coin}_stationary_text_data.parquet.gzip')
        targets = pd.read_parquet(f'../../2_data_processing/numeric_data/{coin}_targets.parquet.gzip')

        # combine and clean data, add `timestep` column
        if coin == 'btc':
            data = pd.concat([numeric_data, nlp_data], axis=1).loc[1314662400:1678752000]
        elif coin == 'eth':
            data = pd.concat([numeric_data, nlp_data], axis=1).loc[1445472000:1678838400]
        data = data.fillna(method='ffill').fillna(0).replace([np.inf, -np.inf], 0)

        # select only the desired explanatory variables
        if args.mode == 'stationary_no_nlp':
            columns_to_keep = [
                col for col in data.columns
                if 'bart_mnli' not in col and 'roberta' not in col
            ]
            X = data[columns_to_keep]
        elif args.mode == 'stationary_twitter_roberta':
            columns_to_keep = [
                col for col in data.columns
                if 'finetuned' not in col and 'bart_mnli' not in col
            ]
            X = data[columns_to_keep]
        elif args.mode == 'stationary_bart_mnli':
            columns_to_keep = [
                col for col in data.columns if 'roberta' not in col
            ]
            X = data[columns_to_keep]
        elif args.mode == 'stationary_nlp_pretrained':
            columns_to_keep = [
                col for col in data.columns if 'finetuned' not in col
            ]
            X = data[columns_to_keep]
        elif args.mode == 'stationary_nlp_finetuned':
            columns_to_keep = [
                col for col in data.columns
                if 'bart_mnli' not in col and 'roberta_pretrained' not in col
            ]
            X = data[columns_to_keep]
        elif args.mode == 'stationary_full_data':
            X = data

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
                
                trainer_args = {
                    'max_epochs': trial.suggest_int(
                        'max_epochs', 10, 200
                    ),
                    'gradient_clip_val': trial.suggest_float(
                        'gradient_clip_val', 0.1, 1.0
                    ),
                    'limit_train_batches': trial.suggest_float(
                        'limit_train_batches', 0.8, 1.0
                    )
                }
                model_args = {
                    'learning_rate': trial.suggest_float(
                        'learning_rate', 1e-4, 1e-1, log=True
                    ),
                    'hidden_size': trial.suggest_categorical(
                        'hidden_size', [16, 32, 64, 128]
                    ),
                    'lstm_layers': trial.suggest_int(
                        'lstm_layers', 1, 3
                    ),
                    'attention_head_size': trial.suggest_categorical(
                        'attention_head_size', [4, 8, 16]
                    ),
                    'dropout': trial.suggest_float(
                        'dropout', 0.1, 0.5
                    ),
                    'hidden_continuous_size': trial.suggest_categorical(
                        'hidden_continuous_size', [16, 32, 64]
                    ),
                    'optimizer': trial.suggest_categorical(
                        'optimizer', ['Adam', 'RMSprop', 'SGD', 'Adagrad', 'Ranger']
                    ),
                    'reduce_on_plateau_patience': trial.suggest_categorical(
                        'reduce_on_plateau_patience', [5, 10, 15]
                    ),
                }
                batch_size = trial.suggest_categorical(
                    'batch_size', [16, 32, 64, 128]
                )
            
                log_config = {
                    'trial_number': trial.number,
                    'batch_size': batch_size,
                    **trainer_args,
                    **model_args,
                    **cv_config,
                }
                wandb.init(
                    project=f'Crypto TFT {args.mode}',
                    config=log_config,
                    group=STUDY_NAME,
                    reinit=True,
                )
                tft_args = ({
                    'trainer_args': trainer_args,
                    'model_args': model_args,
                    'batch_size': 16,
                    'gpus': args.gpus,
                })
                if problem == 'classification':
                    tft_args['binary_classification'] = True
                    
                result = crossvalidate_movement_tft(
                    tft_args,
                    problem,
                    X,
                    y,
                    price_data,
                    args.folds,
                    **cv_config,
                    wandb_log=True,
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
                    project=f'Crypto TFT {args.mode}',
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
    Variable: price movement (up or down), model: TFT
        Best hyperparameters:
{best_params}
        Best value: {study.best_value:.2f} % excess return'''
        
            with open(f'optuna_results_{args.mode}.txt', 'a') as f:
                f.write('\n' + output)

        for timeframe in (7, 14, 21):

            cv_config['cutoff_tuning'] = True
            cv_config['tuning_method'] = 'accuracy'

            # select only the desired explanatory variables
            if args.mode == 'stationary_no_nlp':
                columns_to_keep = [
                    col for col in data.columns
                    if 'bart_mnli' not in col and 'roberta' not in col
                ]
                X_min = data[columns_to_keep]
                X_max = data[columns_to_keep]
            elif args.mode == 'stationary_twitter_roberta':
                columns_to_keep = [
                    col for col in data.columns
                    if 'finetuned' not in col and 'bart_mnli' not in col
                ]
                X_min = data[columns_to_keep]
                X_max = data[columns_to_keep]
            elif args.mode == 'stationary_bart_mnli':
                columns_to_keep = [
                    col for col in data.columns if 'roberta' not in col
                ]
                X_min = data[columns_to_keep]
                X_max = data[columns_to_keep]
            elif args.mode == 'stationary_nlp_pretrained':
                columns_to_keep = [
                    col for col in data.columns if 'finetuned' not in col
                ]
                X_min = data[columns_to_keep]
                X_max = data[columns_to_keep]
            elif args.mode == 'stationary_nlp_finetuned':
                columns_to_keep = [
                    col for col in data.columns
                    if 'bart_mnli' not in col and 'roberta_pretrained' not in col
                ]
                X_min = data[columns_to_keep]
                X_max = data[columns_to_keep]
            elif args.mode == 'stationary_full_data':
                X_min = data
                X_max = data

            # select targets
            min_target = f'{coin}_price_min_{timeframe}d'
            max_target = f'{coin}_price_max_{timeframe}d'
            y_min = targets[min_target]
            y_max = targets[max_target]

            print(f'\nRunning optimisation for {coin} {timeframe} days extrema...\n')

            # Define the objective function for optuna
            STUDY_NAME = f'{coin}_extrema_{timeframe}d_classification'
            def objective(trial):
                
                trainer_args = {
                    'max_epochs': trial.suggest_int(
                        'max_epochs', 10, 200
                    ),
                    'gradient_clip_val': trial.suggest_float(
                        'gradient_clip_val', 0.1, 1.0
                    ),
                    'limit_train_batches': trial.suggest_float(
                        'limit_train_batches', 0.8, 1.0
                    )
                }
                model_args = {
                    'learning_rate': trial.suggest_float(
                        'learning_rate', 1e-4, 1e-1, log=True
                    ),
                    'hidden_size': trial.suggest_categorical(
                        'hidden_size', [16, 32, 64, 128]
                    ),
                    'lstm_layers': trial.suggest_int(
                        'lstm_layers', 1, 3
                    ),
                    'attention_head_size': trial.suggest_categorical(
                        'attention_head_size', [4, 8, 16]
                    ),
                    'dropout': trial.suggest_float(
                        'dropout', 0.1, 0.5
                    ),
                    'hidden_continuous_size': trial.suggest_categorical(
                        'hidden_continuous_size', [16, 32, 64]
                    ),
                    'optimizer': trial.suggest_categorical(
                        'optimizer', ['Adam', 'RMSprop', 'SGD', 'Adagrad', 'Ranger']
                    ),
                    'reduce_on_plateau_patience': trial.suggest_categorical(
                        'reduce_on_plateau_patience', [5, 10, 15]
                    ),
                }
                batch_size = trial.suggest_categorical(
                    'batch_size', [16, 32, 64, 128]
                )

                log_config = {
                    'trial_number': trial.number,
                    'batch_size': batch_size,
                    **trainer_args,
                    **model_args,
                    **cv_config,
                }
                wandb.init(
                    project=f'Crypto TFT {args.mode}',
                    config=log_config,
                    group=STUDY_NAME,
                    reinit=True,
                )
                tft_args = ({
                    'trainer_args': trainer_args,
                    'model_args': model_args,
                    'batch_size': 16,
                    'gpus': args.gpus,
                    'binary_classification': True,
                    'positive_class_weight': trial.suggest_int(
                        'positive_class_weight', 1, 20,
                    ),
                })
                result = crossvalidate_extrema_tft(
                    tft_args,
                    X_min,
                    X_max,
                    y_min,
                    y_max,
                    price_data,
                    args.folds,
                    **cv_config,
                    wandb_log=True,
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
                    project=f'Crypto TFT {args.mode}',
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
    Variable: extrema (min/max), +/- {timeframe} days, model: TFT

        Best hyperparameters:
{best_params}
        Best value: {study.best_value:.2f} % excess return'''

            with open(f'optuna_results_{args.mode}.txt', 'a') as f:
                f.write('\n' + output)

        with open(f'optuna_results_{args.mode}.txt', 'a') as f:
            f.write('\n'*2)

    print(f'Done! Hyperparameter optimisation results saved as `optuna_results_{args.mode}.txt`.')
    

if __name__=='__main__':
    
    hyperparameter_opt_tft()
