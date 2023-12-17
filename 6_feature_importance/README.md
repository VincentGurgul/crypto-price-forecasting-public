# Feature Importance Analysis

This directory contains scripts for the assessment of the feature importance using an XGBoost model trained on daily price fluctuations.


## Contents

```
├── btc_causal_variables.txt: Variables that have a causal relationship with the daily price fluctuations of Bitcoin.
├── config.py: Optimal configuration settings for XGBoost determined during hyperparameter optimisation.
├── eth_causal_variables.txt: Variables that have a causal relationship with the daily price fluctuations of Ethereum.
├── main.py: The main script used to extract and analyse feature importances from XGBoost.
├── xgb_importances_aggregated.txt: Feature importances aggregated by feature category.
└── xgb_importances_disaggregated.txt: Disaggregated feature importances (aggregated only for all lags of each variable).
```