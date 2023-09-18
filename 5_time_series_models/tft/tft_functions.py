''' Functions for training and evaluating Temporal Fusion Transformer with GPU
utilisation on crpyto dataset.

Note: Because there has been a change in how pytorch lightning is imported one
needs to make a change in the pytorch_forecasting source code to make this
script work:
- Go to the folder of your python version, e.g. /python3.9/
- Open the file /site-packages/pytorch_forecasting/models/base_models.py
- Replace all imports `lightning.pytorch` with `pytorch_lightning`
'''

import sys
sys.path += ['../']

import warnings
import os
import torch
import wandb

import numpy as np
import pandas as pd
import pytorch_lightning as pl

from tqdm import tqdm
from copy import deepcopy
from sklearn import metrics
from sklearn.model_selection import TimeSeriesSplit
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import RMSE, convert_torchmetric_to_pytorch_forecasting_metric

from functions import get_cutoffs, get_max_profit_cutoff, calculate_profit

warnings.filterwarnings('ignore')


class TFTModel():
    
    def __init__(self,
                 trainer_args: dict,
                 model_args: dict,
                 batch_size: int,
                 gpus: int,
                 max_encoder_length: int = 14,
                 binary_classification: bool = False,
                 positive_class_weight: int = None,
                 seed: int = 42):
        ''' Class for training and predicting one step ahead with the Temporal
        Fusion Transformer on a time series dataset.
        
        Args:
            trainer_args (dict): Dictionary of keyword arguments for the
                pytorch_lightning.Trainer class
            model_args (dict): Dictionary of keyword arguments for the
                pytorch_forecasting.TemporalFusionTransformer class
            batch_size (int): Batch size for the torch.DataLoader
            gpus (int): Number of GPUs in use, can be either 1 or 2
            max_encoder_length (int, optional): Maximum lookback period of the 
                Temporal Fusion Transformer in timesteps, defaults to 14
            binary_classification (bool, optional): Whether the task is a
                binary classification problem, defaults to False.
            positive_class_weight (int, optional): Class weights of the positive
                class for binary classification problems. Defaults to None, i.e.
                equal weights.
            seed (int, optional): Seed for pytorch_lightning, defaults to 42
        '''
        self.trainer_args = trainer_args
        self.model_args = model_args
        self.batch_size = batch_size
        self.gpus = gpus
        self.max_encoder_length = max_encoder_length
        self.binary_classification = binary_classification
        self.seed = seed
        
        # Define model and loss
        if binary_classification:
            if positive_class_weight:
                positive_class_weight = torch.tensor([positive_class_weight])
            self.loss = convert_torchmetric_to_pytorch_forecasting_metric(
                torch.nn.BCEWithLogitsLoss(pos_weight=positive_class_weight)
            )
        else:
            self.loss = RMSE()

    def fit(self, X: pd.DataFrame, y: pd.Series):
        ''' Function for training the Temporal Fusion Transformer on a given
        time series dataset.
        
        Args:
            X (pd.DataFrame): Dataframe containing explanatory time
                series and a column called `timestep`
            y (pd.Series): Series containg the target time series

        Raises:
            AssertionError: If X and y do not have the same number of rows
            AssertionError: If target is not numeric, since TFT requires floats
                for training
        '''
        assert len(X) == len(y)
        assert pd.api.types.is_numeric_dtype(y)

        y = y.astype(float) # TFT requires floats for training

        # TimeSeriesDataset does not allow for dots in column names
        X.columns = X.columns.str.replace('.', '_', regex=True)
        train_df = pd.concat([X, y], axis=1) # merge data

        # extract variable names
        self.target_var = y.name
        self.variables = list(X.columns)

        # create column `constant` with constant group id, since we're woking
        # with a time series and not a panel dataset
        train_df['constant'] = 1

        # reset index to have two different columns indexing time, the UTC 
        # timestamp (seconds since 1970) and the timestep relative to the 
        # beginning of the dataset (days since first entry)
        train_df = train_df.reset_index(names='timestamp')
        train_df = train_df.reset_index(names='timestep')

        # save last timestep to continue counting there with the test set
        # indexing
        self.last_timestep = train_df.timestep.iloc[-1]

        # create a time series dataset
        self.max_prediction_length = 1 # one step ahead forecasting
        train_cutoff = train_df['timestep'].max() - self.max_prediction_length

        training = TimeSeriesDataSet(
            train_df[lambda x: x['timestep'] <= train_cutoff],
            time_idx='timestep',
            target=self.target_var,
            group_ids=['constant'], 
            min_encoder_length=0,
            max_encoder_length=self.max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=self.max_prediction_length,
            time_varying_known_categoricals=[],  
            time_varying_known_reals=['timestamp'],
            time_varying_unknown_categoricals=[],
            time_varying_unknown_reals=self.variables,
            add_relative_time_idx=False,
            add_target_scales=False,
            add_encoder_length=False,
            allow_missing_timesteps=False,
        )

        # create dataloaders for model
        train_dataloader = training.to_dataloader(train=True,
                                                  batch_size=self.batch_size,
                                                  num_workers=40)

        # initalise trainer and model
        pl.seed_everything(self.seed)
        trainer = pl.Trainer(
            **self.trainer_args,
            enable_model_summary=True,
        )
        self.tft = TemporalFusionTransformer.from_dataset(
            training,
            **self.model_args,
            loss=self.loss,
            log_interval=50,
        )
        for param in self.tft.parameters():
            assert param.requires_grad, "Some model parameters don't require gradients."


        # fit network to training data
        trainer.fit(
            model=self.tft,
            train_dataloaders=train_dataloader,
        )

        return self

    def predict(self, X: pd.DataFrame, y: pd.Series) -> np.array:
        ''' Function for predicting with the trained Temporal Fusion Transformer.
        
        Args:
            X (pd.DataFrame): Dataframe containing explanatory time
                series and a column called `timestep` whose numbering starts
                where the numbering of the training data ended
            y (pd.Series): Series containing the target time series (required
                since the model is autoregressive and needs past lags of the 
                target for prediction too)
        
        Returns:
            np.array[float]: one-step-ahead predictions for each individual
            
        Raises:
            AttributeError: If TFT was not fit prior to predicting
            Error: If number of variables or their dtypes are not equal to those
                in the training data
        '''
        if 'tft' not in dir(self):
            raise AttributeError('Temporal Fusion Transformer not yet trained. Please first run TFTRegressor.fit()')
        
        # TimeSeriesDataset does not allow for dots in column names
        X.columns = X.columns.str.replace('.', '_', regex=True)
        test_df = pd.concat([X, y], axis=1) # merge data

        # create column `constant` with constant group id, since we're working
        # with a time series and not a cross-sectional dataset
        test_df['constant'] = 1

        # reset index to have two different columns indexing time, the UTC 
        # timestamp (seconds since 1970) and the timestep relative to the 
        # beginning of the dataset (days since first entry)
        test_df = test_df.reset_index(names='timestamp')
        test_df = test_df.reset_index(names='timestep')

        test_df['timestep'] += self.last_timestep + 1

        # create time series dataset
        testing = TimeSeriesDataSet(
            test_df,
            time_idx='timestep',
            target=self.target_var,
            group_ids=['constant'],
            min_encoder_length=0,
            max_encoder_length=self.max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=self.max_prediction_length,
            time_varying_known_categoricals=[],  
            time_varying_known_reals=['timestamp'],
            time_varying_unknown_categoricals=[],
            time_varying_unknown_reals=self.variables,
            add_relative_time_idx=False,
            add_target_scales=False,
            add_encoder_length=False,
            allow_missing_timesteps=False,
        )

        # calculate predictions on time series dataset
        predictions = self.tft.predict(testing,
                                       return_x=False,
                                       return_index=True)

        # save calculations of each GPU separately
        predictions = pd.DataFrame(predictions[0].cpu().numpy(),
                                   index=predictions[2].timestep)

        if self.gpus == 2:
            # run try and except loop so that final results are only saved by one
            # GPU, if two GPUs are enabled and running calculations in parallel
            try:
                predictions.to_csv(f'predictions_{torch.cuda.current_device()}.csv')
                predictions = pd.concat([
                    pd.read_csv('predictions_0.csv'),
                    pd.read_csv('predictions_1.csv'),
                ])
                os.remove('predictions_0.csv')
                os.remove('predictions_1.csv')
                predictions = (predictions
                               .set_index('timestep')
                               .sort_index()
                               .iloc[:-self.max_encoder_length])
                predictions.index = test_df.timestamp
            except:
                pass

        elif self.gpus == 1:
            predictions = (predictions
                           .sort_index()
                           .iloc[:-self.max_encoder_length])
            predictions.index = test_df.timestamp
        
        if self.binary_classification:
            def sigmoid(x):
                return 1.0 / (1.0 + np.exp(-x))
            return sigmoid(predictions.squeeze())
        else:
            return predictions.squeeze()


def crossvalidate_movement_tft(tft_args: dict,
                               problem: str,
                               X,
                               y,
                               price_data,
                               folds: int = 7,
                               add_constant_sliding_window: bool = True,
                               constant_window_size: int = 1000,
                               cutoff_tuning: bool = True,
                               tuning_method: str = 'accuracy',
                               init_state: str = 'invested',
                               skip_first: bool = False,
                               wandb_log: bool = False,
                               **kwargs) -> np.array:
    ''' Runs two types of time series cross validation for cryptocurrency price
    movement prediction models and reports mean results. First time series CV
    with an increasing sliding window is run. Second a constant sliding window
    is applied. In each case the data is split into `folds` folds.
    
    Args:
        tft_args (dict): Keyword arguments for the TFTRegressor class
        problem (str): Type of prediction problem, either `regression` or
            `classification`.
        X (pd.Dataframe): explanatory variables
        y (pd.Dataframe): target variable
        price_data (pd.Series or pd.Dataframe): price movements (log price difference)
            of cryptocurrency used for profit calculation
        folds (int, optional): Number of folds for each type of cross validation,
            defaults to 7
        add_constant_sliding_window (bool, optional): Whether to also run cross
            validation with a constant sliding window approach. Defaults to
            True.
        cutoff_tuning (bool, optional): Whether to predict using the scikit
            method `predict_proba` which provides continuous predictions and
            empirically tune the cutoff or whether to use the scikit method
            `predict` which provides discrete predictions. Defaults to False.
        init_state (str, optional): Initial state for profit calculation. Either
            `invested` or `disinvested`. If `invested` coin is bought at time
            zero, otherwise it is only bought once a buy signal occurs.
            Defaults to `invested`.
        skip_first (bool, optional): Whether to skip the first fold in the cross
            validation
        wandb_log (bool, optional): Whether to log results to Weights & Biases.
            Requires logging in with >>> wandb login. Defaults to False.
        **kwargs: additional keyword arguments for the function
            sklearn.model_selection.TimeSeriesSplit()
        
    Returns:
        np.array: (
            mean buy and hold (baseline) profit;
            mean prediction profit;
            mean number of trades using predictions;
            mean excess profit, i.e. difference of prediction profit and
                baseline profit;
            mean maximum achievable profit with given target;
            mean trades corresponding to the max profit;
            mean prediction AUC ROC;
            mean prediction accuracy;
            (only if cutoff tuning enabled) mean cutoff;
        )

    Raises:
        IndexError: If X and y do not have the same number of rows
        TypeError: If target is not numeric
    '''
    if not len(X) == len(y):
        raise IndexError('X and y lengths must match')
    if not pd.api.types.is_numeric_dtype(y):
        raise TypeError('y must be numeric')

    y_discrete = (y > 0) * 1
    y_discrete = pd.Series(y_discrete, index=y.index)
    
    if problem == 'classification':
        y = y_discrete

    results = []
    
    if skip_first:
        folds += 1

    df = pd.concat([X, y], axis=1).reset_index(drop=True)
    df = df.reset_index(names='timestep')
    df.columns = df.columns.str.replace('.', '_', regex=True)

    # increasing sliding window
    tss = TimeSeriesSplit(n_splits=folds, **kwargs).split(X)
    for fold, (train_index, test_index) in tqdm(enumerate(tss)):

        if skip_first and fold == 0:
            continue

        # predict on split
        model = deepcopy(TFTModel(**tft_args))
        model.fit(X.iloc[train_index], y.iloc[train_index])
        preds = model.predict(X.iloc[test_index], y.iloc[test_index])
        preds = pd.Series(preds, index=X.iloc[test_index].index)

        # discretise predictions
        if problem == 'classification' and cutoff_tuning and tuning_method != 'profit':
            cutoffs = get_cutoffs(y.iloc[test_index], preds, plot=False)
            if tuning_method == 'accuracy':
                cutoff = cutoffs[0]
            elif tuning_method == 'youden':
                cutoff = cutoffs[1]
            preds = (preds >= cutoff) * 1
        elif problem == 'classification' and cutoff_tuning and tuning_method == 'profit':
            cutoff = get_max_profit_cutoff(price_data.iloc[test_index],
                                           preds,
                                           y.iloc[test_index],
                                           'movement',
                                           init_state)
            preds = (preds >= cutoff) * 1
        elif problem == 'regression':
            preds = (preds > 0) * 1
            preds = pd.Series(preds, index=X.iloc[test_index].index)

        # calculate AUC ROC and accuracy
        prediction_auc = metrics.roc_auc_score(
            y_discrete.iloc[test_index], preds)
        prediction_acc = metrics.accuracy_score(
            y_discrete.iloc[test_index], preds)

        # calculate profit
        profits = calculate_profit(price_data.iloc[test_index],
                                   preds,
                                   y_discrete,
                                   prediction_type='movement',
                                   init_state=init_state)
        
        if wandb_log:
            wandb.log(data={'cv_fold_profit': profits[3]}, step=fold)

        # append to list
        try:
            results.append((
                profits[0],
                profits[1],
                profits[2],
                profits[3],
                profits[4],
                profits[5],
                prediction_auc,
                prediction_acc,
                cutoff,
            ))
        except:
            results.append((
                profits[0],
                profits[1],
                profits[2],
                profits[3],
                profits[4],
                profits[5],
                prediction_auc,
                prediction_acc,
            ))

    # constant sliding window
    if add_constant_sliding_window:
        tss = TimeSeriesSplit(
            n_splits=folds, max_train_size=constant_window_size, **kwargs).split(X)
        for fold, (train_index, test_index) in tqdm(enumerate(tss)):
            
            if skip_first and fold == 0:
                continue

            # predict on split
            model = deepcopy(TFTModel(**tft_args))
            model.fit(X.iloc[train_index], y.iloc[train_index])
            preds = model.predict(X.iloc[test_index], y.iloc[test_index])
            preds = pd.Series(preds, index=X.iloc[test_index].index)

            # discretise predictions
            if problem == 'classification' and cutoff_tuning and tuning_method != 'profit':
                cutoffs = get_cutoffs(y.iloc[test_index], preds, plot=False)
                if tuning_method == 'accuracy':
                    cutoff = cutoffs[0]
                elif tuning_method == 'youden':
                    cutoff = cutoffs[1]
                preds = (preds >= cutoff) * 1
            elif problem == 'classification' and cutoff_tuning and tuning_method == 'profit':
                cutoff = get_max_profit_cutoff(price_data.iloc[test_index],
                                            preds,
                                            y.iloc[test_index],
                                            'movement',
                                            init_state)
                preds = (preds >= cutoff) * 1
            elif problem == 'regression':
                preds = (preds > 0) * 1
                preds = pd.Series(preds, index=X.iloc[test_index].index)

            # calculate AUC ROC and accuracy
            prediction_auc = metrics.roc_auc_score(
                y_discrete.iloc[test_index], preds)
            prediction_acc = metrics.accuracy_score(
                y_discrete.iloc[test_index], preds)

            # calculate profit
            profits = calculate_profit(price_data.iloc[test_index],
                                       preds,
                                       y_discrete,
                                       prediction_type='movement',
                                       init_state=init_state)

            # append to list
            try:
                results.append((
                    profits[0],
                    profits[1],
                    profits[2],
                    profits[3],
                    profits[4],
                    profits[5],
                    prediction_auc,
                    prediction_acc,
                    cutoff,
                ))
            except:
                results.append((
                    profits[0],
                    profits[1],
                    profits[2],
                    profits[3],
                    profits[4],
                    profits[5],
                    prediction_auc,
                    prediction_acc,
                ))

    return np.mean(results, axis=0)


def crossvalidate_extrema_tft(tft_args,
                              X_min,
                              X_max,
                              y_min,
                              y_max,
                              price_data,
                              folds: int = 7,
                              add_constant_sliding_window: bool = True,
                              constant_window_size: int = 1000,
                              cutoff_tuning: bool = True,
                              tuning_method: str = 'accuracy',
                              init_state: str = 'invested',
                              skip_first: bool = False,
                              wandb_log: bool = False,
                              **kwargs) -> np.array:
    ''' Runs two types of time series cross validation for cryptocurrency
    extremepoint (minimum / maximum) prediction models and reports mean
    results. First time series CV with an increasing sliding window is run.
    Second a constant sliding window is applied. In each case the data is
    split into `folds` folds.
    
    Args:
        tft_args (dict): Keyword arguments for the TFTRegressor class
        X_min (pd.Dataframe): explanatory variables for minima prediction
        X_max (pd.Dataframe): explanatory variables for maxima prediction
        y_min (pd.Series or pd.Dataframe): target variable for minima prediction
        y_max (pd.Series or pd.Dataframe): target variable for maxima prediction
        price_data (pd.Series or pd.Dataframe): price movements (log price difference)
            of cryptocurrency used for profit calculation
        folds (int, optional): Number of folds for each type of cross validation,
            defaults to 5
        add_constant_sliding_window (bool, optional): Whether to also run cross
            validation with a constant sliding window approach. Defaults to
            True.
        cutoff_tuning (bool, optional): Whether to predict using the scikit
            method `predict_proba` which provides continuous predictions and
            empirically tune the cutoff or whether to use the scikit method
            `predict` which provides discrete predictions. Defaults to False.
        init_state (str, optional): Initial state for profit calculation. Either
            `invested` or `disinvested`. If `invested` coin is bought at time
            zero, otherwise it is only bought once a buy signal occurs.
            Defaults to `invested`.
        skip_first (bool, optional): Whether to skip the first fold in the cross
            validation
        wandb_log (bool, optional): Whether to log results to Weights & Biases.
            Requires logging in with >>> wandb login. Defaults to False.
        **kwargs: additional keyword arguments for the function
            sklearn.model_selection.TimeSeriesSplit()
        
    Returns:
        np.array: (
            mean buy and hold (baseline) profit;
            mean prediction profit;
            mean number of trades using predictions;
            mean excess profit, i.e. difference of prediction profit and
                baseline profit;
            mean maximum achievable profit with given target;
            mean trades corresponding to the max profit;
            mean minima prediction AUC ROC;
            mean maxima prediction AUC ROC;
            mean minima prediction accuracy;
            mean maxima prediction accuracy;
            (only if cutoff tuning enabled) mean minima cutoff;
            (only if cutoff tuning enabled) mean maxima cutoff;
        )
    Raises:
        IndexError: If the input dataframes do not all have the same number
            of rows
        TypeError: If targets are not numeric
    '''
    if not len(X_min) == len(y_min) == len(X_max) == len(y_max):
        raise IndexError('Input dataframe lengths must match.')
    if not pd.api.types.is_numeric_dtype(y_min) and pd.api.types.is_numeric_dtype(y_max):
        raise TypeError('Targets must be numeric.')

    results = []
    
    if skip_first:
        folds += 1

    # increasing sliding window
    tss = TimeSeriesSplit(n_splits=folds, **kwargs).split(X_min)
    for fold, (train_index, test_index) in tqdm(enumerate(tss)):
        
        if skip_first and fold == 0:
            continue

        # fit models and predict
        min_model = deepcopy(TFTModel(**tft_args))
        min_model = min_model.fit(X_min.iloc[train_index],
                                  y_min.iloc[train_index])
        y_pred_min = min_model.predict(X_min.iloc[test_index],
                                       y_min.iloc[test_index])
        y_pred_min = pd.Series(y_pred_min, index=X_min.iloc[test_index].index)

        max_model = deepcopy(TFTModel(**tft_args))
        max_model = max_model.fit(X_max.iloc[train_index],
                                  y_max.iloc[train_index])
        y_pred_max = max_model.predict(X_max.iloc[test_index],
                                       y_max.iloc[test_index])
        y_pred_max = pd.Series(y_pred_max, index=X_max.iloc[test_index].index)
        
        # calculate AUC ROC
        min_prediction_auc = metrics.roc_auc_score(
            y_min.iloc[test_index], y_pred_min)
        max_prediction_auc = metrics.roc_auc_score(
            y_max.iloc[test_index], y_pred_max)
        
        # discretise continuous predictions
        if cutoff_tuning and tuning_method != 'profit':
            min_cutoffs = get_cutoffs(y_min.iloc[test_index], y_pred_min, plot=False)
            if tuning_method == 'accuracy':
                min_cutoff = min_cutoffs[0]
            elif tuning_method == 'youden':
                min_cutoff = min_cutoffs[1]
            y_pred_min = (y_pred_min >= min_cutoff) * 1
            max_cutoffs = get_cutoffs(y_max.iloc[test_index], y_pred_max, plot=False)
            if tuning_method == 'accuracy':
                max_cutoff = max_cutoffs[0]
            elif tuning_method == 'youden':
                max_cutoff = max_cutoffs[1]
            y_pred_max = (y_pred_max >= max_cutoff) * 1
        elif cutoff_tuning and tuning_method == 'profit':
            y_pred_combined = pd.DataFrame({
                'min': y_pred_min,
                'max': y_pred_max,
            })
            y_true_combined = pd.DataFrame({
                'min': y_min.iloc[test_index],
                'max': y_max.iloc[test_index],
            })
            cutoff = get_max_profit_cutoff(price_data.iloc[test_index],
                                           y_pred_combined,
                                           y_true_combined,
                                           'extrema',
                                           init_state)
            y_pred_min = (y_pred_min >= cutoff) * 1
            y_pred_max = (y_pred_max >= cutoff) * 1
            min_cutoff = cutoff
            max_cutoff = cutoff

        # calculate accuracy
        min_prediction_acc = metrics.accuracy_score(
            y_min.iloc[test_index], y_pred_min)
        max_prediction_acc = metrics.accuracy_score(
            y_max.iloc[test_index], y_pred_max)

        # concatinate min and max predictions
        y_pred_discrete = pd.DataFrame({
            'min': y_pred_min,
            'max': y_pred_max,
        })
        y_true_discrete = pd.DataFrame({
            'min': y_min.iloc[test_index],
            'max': y_max.iloc[test_index],
        })

        # calculate profit
        profits = calculate_profit(price_data.iloc[test_index],
                                   y_pred_discrete,
                                   y_true_discrete,
                                   prediction_type='extrema',
                                   init_state=init_state)

        if wandb_log:
            wandb.log(data={'cv_fold_profit': profits[3]}, step=fold)

        # append to list
        try:
            results.append((
                profits[0],
                profits[1],
                profits[2],
                profits[3],
                profits[4],
                profits[5],
                min_prediction_auc,
                max_prediction_auc,
                min_prediction_acc,
                max_prediction_acc,
                min_cutoff,
                max_cutoff,
            ))
        except:
            results.append((
                profits[0],
                profits[1],
                profits[2],
                profits[3],
                profits[4],
                profits[5],
                min_prediction_auc,
                max_prediction_auc,
                min_prediction_acc,
                max_prediction_acc,
            ))

    # constant sliding window
    if add_constant_sliding_window:
        tss = TimeSeriesSplit(
            n_splits=folds, max_train_size=constant_window_size, **kwargs).split(X_min)
        for fold, (train_index, test_index) in tqdm(enumerate(tss)):
            
            if skip_first and fold == 0:
                continue

            # fit models and predict
            min_model = deepcopy(TFTModel(**tft_args))
            min_model = min_model.fit(X_min.iloc[train_index],
                                    y_min.iloc[train_index])
            y_pred_min = min_model.predict(X_min.iloc[test_index],
                                        y_min.iloc[test_index])
            y_pred_min = pd.Series(y_pred_min, index=X_min.iloc[test_index].index)

            max_model = deepcopy(TFTModel(**tft_args))
            max_model = max_model.fit(X_max.iloc[train_index],
                                    y_max.iloc[train_index])
            y_pred_max = max_model.predict(X_max.iloc[test_index],
                                        y_max.iloc[test_index])
            y_pred_max = pd.Series(y_pred_max, index=X_max.iloc[test_index].index)
            
            # calculate AUC ROC
            min_prediction_auc = metrics.roc_auc_score(
                y_min.iloc[test_index], y_pred_min)
            max_prediction_auc = metrics.roc_auc_score(
                y_max.iloc[test_index], y_pred_max)
            
            # discretise continuous predictions
            if cutoff_tuning and tuning_method != 'profit':
                min_cutoffs = get_cutoffs(y_min.iloc[test_index], y_pred_min, plot=False)
                if tuning_method == 'accuracy':
                    min_cutoff = min_cutoffs[0]
                elif tuning_method == 'youden':
                    min_cutoff = min_cutoffs[1]
                y_pred_min = (y_pred_min >= min_cutoff) * 1
                max_cutoffs = get_cutoffs(y_max.iloc[test_index], y_pred_max, plot=False)
                if tuning_method == 'accuracy':
                    max_cutoff = max_cutoffs[0]
                elif tuning_method == 'youden':
                    max_cutoff = max_cutoffs[1]
                y_pred_max = (y_pred_max >= max_cutoff) * 1
            elif cutoff_tuning and tuning_method == 'profit':
                y_pred_combined = pd.DataFrame({
                    'min': y_pred_min,
                    'max': y_pred_max,
                })
                y_true_combined = pd.DataFrame({
                    'min': y_min.iloc[test_index],
                    'max': y_max.iloc[test_index],
                })
                cutoff = get_max_profit_cutoff(price_data.iloc[test_index],
                                            y_pred_combined,
                                            y_true_combined,
                                            'extrema',
                                            init_state)
                y_pred_min = (y_pred_min >= cutoff) * 1
                y_pred_max = (y_pred_max >= cutoff) * 1
                min_cutoff = cutoff
                max_cutoff = cutoff

            # calculate accuracy
            min_prediction_acc = metrics.accuracy_score(
                y_min.iloc[test_index], y_pred_min)
            max_prediction_acc = metrics.accuracy_score(
                y_max.iloc[test_index], y_pred_max)

            # concatinate min and max predictions
            y_pred_discrete = pd.DataFrame({
                'min': y_pred_min,
                'max': y_pred_max,
            })
            y_true_discrete = pd.DataFrame({
                'min': y_min.iloc[test_index],
                'max': y_max.iloc[test_index],
            })

            # calculate profit
            profits = calculate_profit(price_data.iloc[test_index],
                                    y_pred_discrete,
                                    y_true_discrete,
                                    prediction_type='extrema',
                                    init_state=init_state)

            if wandb_log:
                wandb.log(data={'cv_fold_profit': profits[3]}, step=fold)

            # append to list
            try:
                results.append((
                    profits[0],
                    profits[1],
                    profits[2],
                    profits[3],
                    profits[4],
                    profits[5],
                    min_prediction_auc,
                    max_prediction_auc,
                    min_prediction_acc,
                    max_prediction_acc,
                    min_cutoff,
                    max_cutoff,
                ))
            except:
                results.append((
                    profits[0],
                    profits[1],
                    profits[2],
                    profits[3],
                    profits[4],
                    profits[5],
                    min_prediction_auc,
                    max_prediction_auc,
                    min_prediction_acc,
                    max_prediction_acc,
                ))

    return np.mean(results, axis=0)
