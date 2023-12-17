''' Regularisation hyperparameter optimisation for crypto price forecasting with OLS. '''

import sys
sys.path += ['../../', '../../../']

import re
import numpy as np
import pandas as pd
import json
import optuna
import wandb

from argparse import ArgumentParser
from sklearn.linear_model import Ridge
from functions import *

from utils.wrappers import timeit, telegram_notify


@timeit
@telegram_notify
def ols_regularisation_opt():
    
    modes = [
        'causal_stationary_vader',
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
        description='Crypto price forecasting with Lasso/Ridge OLS'
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
        help='(int) number of folds for each time series cross validation, default = 7',
    )
    parser.add_argument(
        '-i',
        '--iter',
        type=int,
        default=200,
        help='(int) number of trial runs per optuna run, i.e. per penalty type, default = 50',
    )
    args = parser.parse_args()
    
    print(f'\n{args}\n')
    
    if args.mode not in modes:
        raise ValueError(f'Selected mode `{args.mode}` is invalid. Please select one of the following modes: {modes}')

    # Print args to results file
    with open(f'optuna_regularisation_results_{args.mode}.txt', 'w') as f:
        f.write(f'''Args:
    Optimising profit based on time series cross validation with {args.folds*2} folds.
    Optuna running {args.iter} trial runs per model.\n\n''')
    
    # Set model configuration that isn't up for optimisation
    base_config = {
        'max_iter': 1000,
        'random_state': 42,
    }
    
    # Set range for hyperparameter optimisation
    alpha_config = {
        'name': 'alpha',
        'low': 1e-3,
        'high': 100,
    }
    solver_config = {
        'name': 'solver',
        'choices': ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
    }
    
    for coin in ('btc', 'eth'):
        
        # specify configuration for cross validation
        cv_config = {
            'add_constant_sliding_window': False,
            'constant_window_size': 1000 if coin == 'btc' else 600,
            'cutoff_tuning': False,
            'skip_first': True,
        }
        
        with open(f'optuna_regularisation_results_{args.mode}.txt', 'a') as f:
            f.write(f'----------------\nCoin: {coin.upper()}\n----------------')
        
        price_data = pd.read_parquet(f'../../../2_data_processing/numeric_data/{coin}_price_data.parquet.gzip').iloc[:,0]
        
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
        
        # Select only the desired explanatory variables
        if args.mode in ['causal_stationary_no_nlp', 'causal_nonstationary']:
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
        elif args.mode == 'causal_stationary_full_data':
            X = data[price_log_difference_vars]
        else:
            columns_to_keep = [
                col for col in data.columns
                if all(i not in col for i in ['bart_mnli', 'roberta', 'vader'])
            ]
            X = data[columns_to_keep]
        
        # Select the target variable
        y = targets[f'{coin}_price_log_difference']

        model = Ridge
        
        print(f'\nRunning optimisation for {coin} price movement {model.__name__} regression...\n')
        
        # Define the objective function for optuna
        STUDY_NAME = f'{coin}_movement_{model.__name__}_regression'
        def objective(trial):
            
            model_config = {
                'alpha': trial.suggest_float(**alpha_config, log=True),
                'solver': trial.suggest_categorical(**solver_config),
            }
                
            log_config = {
                'trial_number': trial.number,
                **model_config,
                **cv_config,
            }
            wandb.init(
                project=f'Crypto Reg-OLS {args.mode}',
                config=log_config,
                group=STUDY_NAME,
                reinit=True,
            )
            result = crossvalidate_movement(
                model(**model_config, **base_config),
                'regression',
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
                project=f'Crypto Reg-OLS {args.mode}',
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
    
        with open(f'optuna_regularisation_results_{args.mode}.txt', 'a') as f:
            f.write('\n' + output + '\n' * 2)

    print(f'Done! Regularisation hyperparameter optimisation results saved as `optuna_regularisation_results_{args.mode}.txt`.')


if __name__=='__main__':
    
    ols_regularisation_opt()
