''' Running Ridge OLS / Logit with optimal hyperparameters for
various Granger causality confidence levels.'''

import numpy as np
from argparse import ArgumentParser
from causality_functions import *


if __name__=='__main__':
    
    # enable number of cross validation folds as argument
    parser = ArgumentParser(
        description='Crypto price forecasting with Random Forest'
    )
    parser.add_argument(
        '-f',
        '--folds',
        type=int,
        default=7,
        help='(int) number of folds for each time series cross validation, defaults to 7',
    )
    parser.add_argument(
        '-d',
        '--double-cv',
        type=bool,
        default=True,
        help='(bool) whether to run cross validation twice, with constant and increasing sliding window, or only once with increasing sliding window CV, defaults to True, i.e. double CV',
    )
    args = parser.parse_args()
    
    # set some global parameters
    base_config = {
        'n_jobs': -1,
        'random_state': 42,
    }

    # Print args to results file
    with open('results.txt', 'w') as f:
        f.write(f'''Args:
    Mean profit based on time series cross validation with {args.folds*2 if args.double_cv else args.folds} folds.
    \n''')
        
    # run all models for various Granger causality confidence levels
    for conf_level in np.linspace(0.01, 0.1, 10):
        
        print(f'\nRunning Granger causality analysis @ {conf_level:.4f} confidence...')
        get_causal_vars(conf_level)
        
        all_target_results = run_all_models(args,
                                            base_config,
                                            True,
                                            'accuracy')
                
        with open('results.txt', 'a') as f:
            f.write(f'''Mean profit at confidence level {conf_level:.4f}:
        {np.mean(all_target_results):.5f} %
        
        By target: {all_target_results}\n\n''')
