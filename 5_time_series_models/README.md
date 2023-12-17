# Time Series Models

This directory is dedicated to training and evaluating the various time series modelling approaches.


## Contents

```
├── lstm
│    ├── hyperparam_opt
│    │    └── hyperparam_opt.py: Script for LSTM hyperparameter optimization.
│    ├── lag_functions.py: Functions related to lag processing.
│    ├── lstm_functions.py: Utility functions specific to LSTM model.
│    └── plot_performance.ipynb: Notebook to plot the performance of LSTM model.
├── mlp
│    ├── hyperparam_opt
│    │    └── hyperparam_opt.py: Script for MLP hyperparameter optimization.
│    └── profit_by_cv_fold
│        └── main.ipynb: Notebook to analyze profit by cross-validation fold.
├── ols
│    └── hyperparam_opt
│         ├── logit_hyperparam_opt.py: Hyperparameter optimization for logit function.
│         └── ols_regularisation_opt.py: Regularization optimization for OLS.
├── plotting
│   ├── plot_crossvalidation.ipynb: Notebook for visualising cross-validation.
│   ├── plotting_functions.py: General plotting utilities.
│   └── summary_barplots_final.ipynb: Notebook for summarising results in bar plots.
├── tft
│    ├── hyperparam_opt.py: Hyperparameter optimization for TFT.
│    └── tft_functions.py: Utility functions specific to TFT model.
├── xgboost
│    ├── hyperparam_opt
│    │    └── hyperparam_opt.py: Script for XGBoost hyperparameter optimization.
│    └── profit_by_cv_fold
│        └── main.ipynb: Notebook to analyze profit by cross-validation fold.
└── functions.py: Common utility functions used across multiple time series models.
```
