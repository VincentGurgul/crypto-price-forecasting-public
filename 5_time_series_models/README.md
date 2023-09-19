# Time Series Models

This directory is dedicated to training and evaluating the various time series modeling approaches for the project.


## Contents

```
.
├── adaboost
│    └── main
│        ├── adb_functions.py: Utility functions for Adaboost model.
│        └── main.py: Main code for implementing the Adaboost model.
├── lstm
│    ├── hyperparam_opt
│    │    └── hyperparam_opt.py: Script for LSTM hyperparameter optimization.
│    ├── lag_functions.py: Functions related to lag processing.
│    ├── lstm_functions.py: Utility functions specific to LSTM model.
│    └── plot_performance.ipynb: Notebook to plot the performance of LSTM model.
├── mlp
│    ├── hyperparam_opt
│    │    └── hyperparam_opt.py: Script for MLP hyperparameter optimization.
│    ├── main
│    │    ├── config.py: Configuration settings for the MLP model.
│    │    ├── main.py: Main code for implementing the MLP model.
│    │    └── mlp_functions.py: Utility functions specific to MLP model.
│    ├── plot_performance.ipynb: Notebook to plot the performance of MLP model.
│    └── profit_by_cv_fold
│        └── main.ipynb: Notebook to analyze profit by cross-validation fold.
├── ols
│    ├── hyperparam_opt
│    │    ├── logit_hyperparam_opt.py: Hyperparameter optimization for logit function.
│    │    └── ols_regularisation_opt.py: Regularization optimization for OLS.
│    ├── main
│    │    ├── config.py: Configuration settings for the OLS model.
│    │    ├── main.py: Main code for implementing the OLS model.
│    │    └── ols_functions.py: Utility functions specific to OLS model.
│    └── plot_performance.ipynb: Notebook to plot the performance of OLS model.
├── random_forest
│    ├── causality_tuning
│    │    ├── causality_functions.py: Functions related to causality tuning.
│    │    ├── main.py: Main script for causality tuning.
│    │    └── results.txt: Results from the causality tuning.
│    ├── hyperparam_opt
│    │    └── hyperparam_opt.py: Script for Random Forest hyperparameter optimization.
│    └── main
│        ├── main.py: Main code for implementing the Random Forest model.
│        └── rf_functions.py: Utility functions specific to Random Forest model.
├── tft
│    ├── hyperparam_opt.py: Hyperparameter optimization for TFT.
│    └── tft_functions.py: Utility functions specific to TFT model.
├── xgboost
│    ├── hyperparam_opt
│    │    └── hyperparam_opt.py: Script for XGBoost hyperparameter optimization.
│    ├── main
│    │    ├── config.py: Configuration settings for the XGBoost model.
│    │    ├── main.py: Main code for implementing the XGBoost model.
│    │    └── xgb_functions.py: Utility functions specific to XGBoost model.
│    ├── plot_performance.ipynb: Notebook to plot the performance of XGBoost model.
│    └── profit_by_cv_fold
│        └── main.ipynb: Notebook to analyze profit by cross-validation fold.
└── functions.py: Common utility functions used across multiple time series models.
```