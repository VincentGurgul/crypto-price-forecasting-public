''' Hyperparameter optimisation for crypto price forecasting with XGBoost. '''

import sys
sys.path += ['../../', '../../../']

import numpy as np
import pandas as pd
import json
import optuna
import wandb

from argparse import ArgumentParser
from xgboost import XGBRegressor, XGBClassifier
from functions import *

from utils.wrappers import timeit, telegram_notify


@timeit
@telegram_notify
def hyperparameter_opt_xgb():
    
    modes = [
        'causal_stationary_nlp_pretrained',
        'causal_stationary_twitter_roberta',
        'causal_stationary_bart_mnli',
        'causal_stationary_nlp_finetuned',
        'causal_stationary_no_nlp',
        'causal_nonstationary',
        'causal_stationary_full_data',
        'full_untransformed',
        'full_stationary',
    ]
    
    # parse cross validation folds and optuna iterations as args
    parser = ArgumentParser(
        description='Crypto price forecasting with XGBoost'
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
        
        price_data = pd.read_parquet(f'../../../2_data_processing/numeric_data/{coin}_price_data.parquet.gzip').iloc[:,0]
        
        with open(f'optuna_results_{args.mode}.txt', 'a') as f:
            f.write(f'----------------\nCoin: {coin.upper()}\n----------------')
        
        if args.mode in ['full_untransformed', 'causal_nonstationary']:
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
        
        if args.mode == 'causal_nonstationary':
            with open(f'../../../4_eda/{coin}_causality_nonstat/{coin}_price_log_difference_causality.txt') as f:
                price_log_difference_vars = f.read().splitlines()
        else:
            with open(f'../../../4_eda/{coin}_causality/{coin}_price_log_difference_causality.txt') as f:
                price_log_difference_vars = f.read().splitlines()
            
        # select only the desired explanatory variables
        if args.mode in ['causal_stationary_no_nlp', 'causal_nonstationary']:
            X = data[price_log_difference_vars]
            columns_to_keep = [
                col for col in X.columns
                if 'bart_mnli' not in col and 'roberta' not in col
            ]
            X = X[columns_to_keep]
        elif args.mode == 'causal_stationary_twitter_roberta':
            X = data[price_log_difference_vars]
            columns_to_keep = [
                col for col in X.columns
                if 'finetuned' not in col and 'bart_mnli' not in col
            ]
            X = X[columns_to_keep]
        elif args.mode == 'causal_stationary_bart_mnli':
            X = data[price_log_difference_vars]
            columns_to_keep = [
                col for col in X.columns if 'roberta' not in col
            ]
            X = X[columns_to_keep]
        elif args.mode == 'causal_stationary_nlp_pretrained':
            X = data[price_log_difference_vars]
            columns_to_keep = [
                col for col in X.columns if 'finetuned' not in col
            ]
            X = X[columns_to_keep]
        elif args.mode == 'causal_stationary_nlp_finetuned':
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

        y = targets[f'{coin}_price_log_difference']

        for problem in ('regression', 'classification'):
            
            # specify model and configuration for cross validation
            if problem == 'regression':
                model = XGBRegressor
            elif problem == 'classification':
                model = XGBClassifier
                cv_config['cutoff_tuning'] = True
                cv_config['tuning_method'] = 'accuracy'
            
            print(f'\nRunning optimisation for {coin} price movement {problem}...\n')
            
            # Define the objective function for optuna
            STUDY_NAME = f'{coin}_movement_{problem}'
            def objective(trial):
                
                model_config = {
                    'n_estimators': trial.suggest_int(
                        'n_estimators', 100, 1400
                    ),
                    'max_depth': trial.suggest_int(
                        'max_depth', 1, 20
                    ),
                    'learning_rate': trial.suggest_float(
                        'learning_rate', 0.01, 0.3, log=True
                    ),
                    'subsample': trial.suggest_float(
                        'subsample', 0.5, 1.0
                    ),
                    'colsample_bytree': trial.suggest_float(
                        'colsample_bytree', 0.5, 1.0
                    ),
                    'reg_alpha': trial.suggest_float(
                        'reg_alpha', 0.001, 1.0, log=True
                    ),
                    'reg_lambda': trial.suggest_float(
                        'reg_lambda', 0.001, 1.0, log=True
                    ),
                    'gamma': trial.suggest_float(
                        'gamma', 0, 1
                    ),
                }
                    
                classifier_obj = model(**model_config)
                
                log_config = {
                    'trial_number': trial.number,
                    **model_config,
                    **cv_config,
                }
                    
                wandb.init(
                    project=f'Crypto XGB {args.mode}',
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
                    project=f'Crypto XGB {args.mode}',
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
        
            with open(f'optuna_results_{args.mode}.txt', 'a') as f:
                f.write('\n' + output)

        for timeframe in (7, 14, 21):
            
            # Set configuration for cross validation
            cv_config['cutoff_tuning'] = True
            cv_config['tuning_method'] = 'accuracy'

            if args.mode == 'causal_nonstationary':
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
            if args.mode in ['causal_stationary_no_nlp', 'causal_nonstationary']:
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
            elif args.mode == 'causal_stationary_twitter_roberta':
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
            elif args.mode == 'causal_stationary_bart_mnli':
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
            elif args.mode == 'causal_stationary_nlp_pretrained':
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
            elif args.mode == 'causal_stationary_nlp_finetuned':
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

            # select targets
            min_target = f'{coin}_price_min_{timeframe}d'
            max_target = f'{coin}_price_max_{timeframe}d'
            y_min = targets[min_target]
            y_max = targets[max_target]

            model = XGBClassifier

            print(f'\nRunning optimisation for {coin} {timeframe} days extrema...\n')

            # Define the objective function for optuna
            STUDY_NAME = f'{coin}_extrema_{timeframe}d_classification'
            def objective(trial):
                
                model_config = {
                    'n_estimators': trial.suggest_int(
                        'n_estimators', 100, 1400
                    ),
                    'max_depth': trial.suggest_int(
                        'max_depth', 1, 20
                    ),
                    'learning_rate': trial.suggest_float(
                        'learning_rate', 0.01, 0.3, log=True
                    ),
                    'subsample': trial.suggest_float(
                        'subsample', 0.5, 1.0
                    ),
                    'colsample_bytree': trial.suggest_float(
                        'colsample_bytree', 0.5, 1.0
                    ),
                    'reg_alpha': trial.suggest_float(
                        'reg_alpha', 0.001, 1.0, log=True
                    ),
                    'reg_lambda': trial.suggest_float(
                        'reg_lambda', 0.001, 1.0, log=True
                    ),
                    'gamma': trial.suggest_float(
                        'gamma', 0, 1
                    ),
                    'scale_pos_weight': trial.suggest_float(
                        'scale_pos_weight', 0, 1
                    ),
                }
                classifier_obj = model(**model_config)

                config = {
                    'trial_number': trial.number,
                    **model_config,
                    **cv_config,
                }
                wandb.init(
                    project=f'Crypto XGB {args.mode}',
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
                    project=f'Crypto XGB {args.mode}',
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

            with open(f'optuna_results_{args.mode}.txt', 'a') as f:
                f.write('\n' + output)

        with open(f'optuna_results_{args.mode}.txt', 'a') as f:
            f.write('\n'*2)

    print(f'Done! Hyperparameter optimisation results saved as `optuna_results_{args.mode}.txt`.')
    

if __name__=='__main__':
    
    hyperparameter_opt_xgb()
