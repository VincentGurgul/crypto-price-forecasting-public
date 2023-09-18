''' Script that extracts feature importances from an XGBoost model trained on
the entire dataset with daily price change as a binary target. '''

import re
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from config import get_model_config


if __name__=='__main__':

    for approach in ('aggregated', 'disaggregated'):
        
        with open(f'xgb_importances_{approach}.txt', 'w') as f:
            f.write(f'XGBoost feature importances by Cryptocurrency ({approach})')

        for coin in ('btc', 'eth'):
            
            with open(f'xgb_importances_{approach}.txt', 'a') as f:
                f.write(f'\n\n----------------\nCoin: {coin.upper()}\n----------------\n\n')
            
            # Import data
            data_path = f'../4_eda/{coin}_stationary_data_lagged.parquet.gzip'
            data = pd.read_parquet(data_path)
            data.columns = data.columns.map('_'.join)
            data = (data.fillna(method='ffill')
                        .fillna(0)
                        .replace([np.inf, -np.inf], 0))

            target_path = f'../2_data_processing/numeric_data/{coin}_targets.parquet.gzip'
            targets = pd.read_parquet(target_path)

            # Select Granger causal variables
            with open(f'{coin}_causal_variables.txt') as f:
                price_log_difference_vars = f.read().splitlines()
                
            X = data[price_log_difference_vars]
            # columns_to_keep = [col for col in X.columns if 'finetuned' not in col]
            # X = X[columns_to_keep]

            # Specify target
            y = targets[f'{coin}_price_log_difference']
            y_discrete = (y > 0) * 1
            y = pd.Series(y_discrete, index=y.index)

            # Fit model
            model_config = get_model_config(coin)
            model = XGBClassifier(**model_config).fit(X, y)
            
            # Access both gain and total gain directly using `get_booster()` method
            gain = model.get_booster().get_score(importance_type='gain')
            total_gain = model.get_booster().get_score(importance_type='total_gain')

            # Pair up feature names with their importances
            summed_average_gains = {}
            summed_total_gains = {}

            # Iterate over feature names in the model
            for feature_name in model.feature_names_in_:

                # Strip coin name, order of integration and lag order
                base_feature_name = re.sub(r'^(btc_|eth_)?|(_d2?_?|\_)?\d{1,2}$', '',
                                           feature_name)

                # Possibly aggregate importance for technical indicators
                if approach == 'aggregated':
                    if '_indicator_' in feature_name:
                        base_feature_name = 'Technical indicators'
                    elif 'volume' in feature_name:
                        base_feature_name = 'Exchange volume data'
                    elif '_balance_' in feature_name or '_transaction_' in feature_name or '_addresses_' in feature_name:
                        base_feature_name = 'Transaction and account balance data'
                    elif '_count' in feature_name or 'followers' in feature_name or 'reddit_subscribers' in feature_name or 'active_48h' in feature_name:
                        base_feature_name = 'Numeric social media data'
                    elif 'subscribers' in feature_name or 'forks' in feature_name or 'stars' in feature_name or 'pull' in feature_name or 'issues' in feature_name or 'additions' in feature_name or 'deletions' in feature_name or 'commit' in feature_name:
                        base_feature_name = 'GitHub metrics'
                    elif 'gtrends' in feature_name:
                        base_feature_name = 'Google Trends'
                    elif 'tweets' in feature_name or '_reddit_' in feature_name or '_news_' in feature_name:
                        base_feature_name = 'NLP data'
                    elif 'difficulty' in feature_name or 'hashrate' in feature_name or 'block' in feature_name or 'supply' in feature_name or 'staking' in feature_name:
                        base_feature_name = 'Technical blockchain metrics'
                    elif '_price_close' in feature_name:
                        base_feature_name = 'Past price data'
                    else:
                        base_feature_name = 'Financial data'

                # Sum the gains and total gains for all lags of each feature
                summed_average_gains[base_feature_name] = (
                    summed_average_gains.get(base_feature_name, 0) +
                    gain.get(feature_name, 0)
                )
                
                summed_total_gains[base_feature_name] = (
                    summed_total_gains.get(base_feature_name, 0) +
                    total_gain.get(feature_name, 0)
                )
                
            # Normalize average and total gains
            total_average_gains = sum(summed_average_gains.values())
            total_total_gains = sum(summed_total_gains.values())

            for key in summed_average_gains:
                summed_average_gains[key] /= total_average_gains
                summed_total_gains[key] /= total_total_gains

            # Create combined dictionary and sort by normalized total gain
            sorted_importances = sorted(summed_average_gains.items(), 
                                        key=lambda x: summed_total_gains[x[0]], 
                                        reverse=True)

            # Save to text file
            with open(f'xgb_importances_{approach}.txt', 'a') as f:
                for name, average_gain in sorted_importances:
                    total_gain = summed_total_gains[name]
                    f.write(f'{name}: total_gain={total_gain}, avg_gain={average_gain}\n')
