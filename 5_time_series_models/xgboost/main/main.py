''' Forecasting crypto prices with XGBoost with optimal hyperparameters. '''

import sys
sys.path += ['../../']

from functions import *
from xgb_functions import run_and_log_xgb


modes = [
    'causal_stationary_nlp_pretrained',
    'causal_stationary_twitter_roberta',
    'causal_stationary_bart_mnli',
    'causal_stationary_nlp_finetuned',
    'causal_stationary_no_nlp',
    'causal_nonstationary',
    # 'causal_stationary_full_data',
    # 'full_untransformed',
    # 'full_stationary',
]

CUTOFF_TUNING = True
TUNING_METHOD = 'accuracy'


if __name__=='__main__':

    args = parse_arguments()
    print(f'\n{args}')
    
    for mode in modes:
        
        print(f'\nRunning analysis for mode: {mode}...\n')
        results_file = f'results_{mode}.txt'

        # Print args to results file
        with open(results_file, 'w') as f:
            f.write(f'''Args:
    Mean profit based on time series cross validation with {args.folds*2 if args.double_cv else args.folds} folds.
    Cutoff tuning: {CUTOFF_TUNING}, tuning method: {TUNING_METHOD}\n\n''')
        
        # Run OLS for each coin
        for coin in ('btc', 'eth'):
            run_and_log_xgb(mode, coin, args, CUTOFF_TUNING, TUNING_METHOD)

        print(f'Done! Results saved as `{results_file}`.')
