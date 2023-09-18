''' Core functions for time series model evaluation. '''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wandb

import argparse
from tqdm import tqdm
from copy import deepcopy
from sklearn import metrics
from sklearn.model_selection import TimeSeriesSplit
from matplotlib.lines import Line2D


def parse_arguments() -> argparse.Namespace:
    ''' Parses arguments with argparse.ArgumentParser. Optional arguments for
    the main time series models functions are:
    
        folds (int, optional): Number of folds for the cross validation,
            defaults to 7
        double-cv (bool, optional): Whether to run cross validation twice,
            with constant and increasing sliding window, or only once with
            increasing sliding window CV, defaults to True, i.e. double CV
            
    Returns:
        Argparse Namespace
    '''
    parser = argparse.ArgumentParser()
    
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
        type=eval,
        default=True,
        help='(bool) whether to run cross validation twice, with constant and increasing sliding window, or only once with increasing sliding window CV, defaults to True, i.e. double CV',
    )
    args = parser.parse_args()
    
    return args


def tpr_fpr_calc(cutoff, yhat_prob, y_true) -> tuple:
    ''' Calculates true positive and false positive rate for continuous
    predictions of a binary target.
    
    Args:
        cutoff (float): cutoff value
        yhat_prob: array of continuous predictions
        y_true: array of binary true targets
        
    Returns:
        (float, float): (tpr, fpr)
    '''
    temp = yhat_prob >= cutoff
    cmat = metrics.confusion_matrix(y_true, temp)
    fpr = cmat[0, 1] / (cmat[0, 1] + cmat[0, 0])
    tpr = cmat[1, 1] / (cmat[1, 1] + cmat[1, 0])
    return tpr, fpr


def get_cutoffs(y_true, yhat_prob) -> tuple:
    ''' Calculates optimal cutoff for continuous predictions of a binary target.
    
    Args:
        y_true: array of binary true targets
        yhat_prob: array of continuous predictions
        
    Returns:
        tuple[float]: (cutoff value that maximises accuracy,
            cutoff value that maximises Youden's J statistic,
            cutoff value that maximises balanced accuracy)
    '''
    accuracies = []
    youden = []
    balanced_accuracies = []
    
    cutoffs = np.arange(0, 1.01, 0.01)

    for cutoff in cutoffs:
        preds = (yhat_prob >= cutoff)
        accuracy = np.sum(preds == y_true) / len(y_true)
        tpr, fpr = tpr_fpr_calc(cutoff, yhat_prob, y_true)
        tnr = 1 - fpr
        balanced_accuracy = (tpr + tnr) / 2
        accuracies.append(accuracy)
        youden.append(tpr-fpr)
        balanced_accuracies.append(balanced_accuracy)

    cutoff_best_acc = cutoffs[np.argmax(accuracies)]
    cutoff_best_youden = cutoffs[np.argmax(youden)]
    cutoff_best_balanced_acc = cutoffs[np.argmax(balanced_accuracies)]
    
    return cutoff_best_acc, cutoff_best_youden, cutoff_best_balanced_acc


def get_max_profit_cutoff(price_data,
                          y_pred,
                          y_true,
                          prediction_type: str,
                          init_state: str = 'invested') -> float:
    ''' Calculates cutoff for continuous predictions of a binary target that
    results in the highest profit.
    
    Args:
        price_data (pd.Series or pd.DataFrame): one-dimensional array of log
            price changes
        y_pred (pd.Series or pd.DataFrame): one or two-dimensional array
            of binary predictions
        y_true (pd.Series or pd.DataFrame): one or two-dimensional array
            of binary true target
        prediction_type (str): `movement` or `extrema`. In the latter case
            two-dimensional predictions are expected, with column names `min`
            and `max`. 
        init_state (str, optional): Initial state for profit calculation. Either
            `invested` or `disinvested`. If `invested` coin is bought at time
            zero, otherwise it is only bought once a buy signal occurs.
            Defaults to `invested`.
            
    Returns:
        float: cutoff value that maximises profit
    '''
    profits = []
    cutoffs = np.arange(0, 1.01, 0.01)

    for cutoff in cutoffs:
        if prediction_type == 'movement':
            y_pred_discrete = (y_pred >= cutoff)
            profit = calculate_profit(price_data,
                                      y_pred_discrete,
                                      y_true,
                                      prediction_type,
                                      init_state)
            profits.append(profit[1])
        elif prediction_type == 'extrema':
            y_pred_discrete = y_pred
            y_pred_discrete['min'] = (y_pred_discrete['min'] >= cutoff)
            y_pred_discrete['max'] = (y_pred_discrete['max'] >= cutoff)
            profit = calculate_profit(price_data,
                                      y_pred_discrete,
                                      y_true,
                                      prediction_type,
                                      init_state)
            profits.append(profit[1])

    return cutoffs[np.argmax(profits)]


def calculate_profit(price_data,
                     y_pred,
                     y_true,
                     prediction_type: str,
                     init_state: str = 'invested') -> tuple:
    ''' Calculates profit of trading strategy on a given timeframe and compares
    it to the profit from a buy and hold strategy on the same time frame.
    
    Args:
        price_data (pd.Series or pd.DataFrame): one-dimensional array of log
            price changes
        y_pred (pd.Series or pd.DataFrame): one or two-dimensional array
            of binary predictions
        y_true (pd.Series or pd.DataFrame): one or two-dimensional array
            of binary true target
        prediction_type (str): `movement` or `extrema`. In the latter case
            two-dimensional predictions are expected, with column names `min`
            and `max`. 
        init_state (str, optional): Initial state for profit calculation. Either
            `invested` or `disinvested`. If `invested` coin is bought at time
            zero, otherwise it is only bought once a buy signal occurs.
            Defaults to `invested`.
        
    Returns:
        tuple(float: buy and hold (baseline) profit; float: prediction profit;
            float: number of trades using predictions; float: excess profit,
            i.e. difference of prediction profit and baseline profit; float:
            maximum achievable profit with given target; float: trades
            corresponding to the max profit)
            
    Raises:
        AssertionError: If price_data and predictions indices do not match.
        AssertionError: If predictions are not binary.
        KeyError: If prediction_type is `extrema` and predictions are not
            two-dimensional with column names `min` and `max`.
    '''
    # check that predictions match price data and that prediction are binary
    assert (price_data.index == y_pred.index).all()
    assert y_pred.isin([0, 1]).all().all()

    # calculate profit under buy and hold strategy
    k = 1
    for i in price_data:
        k *= np.exp(i)
    buy_and_hold_profit = (k - 1) * 100

    # calculate profit for each day using predictions
    # initialise state with init_state (either buy or dont buy on day 1)
    state = init_state
    # initialise number of trades
    if init_state=='invested':
        pred_trades = 1
    elif init_state=='disinvested':
        pred_trades = 0
    profit = 1  # initialise profit with 1, i.e. 0 % profit

    if prediction_type == 'movement':
        for day in price_data.index.astype(int):
            # if holding currency add price change since yesterday to profits
            if state == 'invested':
                profit *= np.exp(price_data[day])
            # if predicting price increase, and not holding, buy currency
            if y_pred[day] == 1:
                if state == 'disinvested':
                    state = 'invested'
                    pred_trades += 1
            # if predicting price decrease, and holding, sell currency
            elif y_pred[day] == 0:
                if state == 'invested':
                    state = 'disinvested'
                    pred_trades += 1

    if prediction_type == 'extrema':
        for day in price_data.index.astype(int):
            # if holding currency add price change since yesterday to profits
            if state == 'invested':
                profit *= np.exp(price_data[day])
            # if predicting minimum, and not holding, buy currency
            if y_pred.loc[day, 'min'] == 1 and y_pred.loc[day, 'max'] == 0:
                if state == 'disinvested':
                    state = 'invested'
                    pred_trades += 1
            # if predicting maximum, and holding, sell currency
            elif y_pred.loc[day, 'min'] == 0 and y_pred.loc[day, 'max'] == 1:
                if state == 'invested':
                    state = 'disinvested'
                    pred_trades += 1

    # add one more trade if currency is held at the end
    if state == 'invested':
        pred_trades += 1
        
    # calculate maximum profit for each day using true target variable
    # initiate state with init_state (either buy or dont buy on day 1)
    state = init_state
    # initialise number of trades
    if init_state=='invested':
        max_trades = 1
    elif init_state=='disinvested':
        max_trades = 0
    max_profit = 1  # initialise profit with 1, i.e. 0 % profit

    if prediction_type == 'movement':
        for day in price_data.index.astype(int):
            # if holding currency add price change since yesterday to profits
            if state == 'invested':
                max_profit *= np.exp(price_data[day])
            # if predicting price increase, buy currency
            if y_true[day] == 1:
                if state == 'disinvested':
                    state = 'invested'
                    max_trades += 1
            # if predicting price decrease, sell currency
            elif y_true[day] == 0:
                if state == 'invested':
                    state = 'disinvested'
                    max_trades += 1

    if prediction_type == 'extrema':
        # check that predictions are two-dimensional
        assert y_true.shape[1] == 2

        for day in price_data.index.astype(int):
            # if holding currency add price change since yesterday to profits
            if state == 'invested':
                max_profit *= np.exp(price_data[day])
            # if predicting minimum, buy currency
            if y_true.loc[day, 'min'] == 1:
                if state == 'disinvested':
                    state = 'invested'
                    max_trades += 1
            # if predicting maximum, sell currency
            elif y_true.loc[day, 'max'] == 1:
                if state == 'invested':
                    state = 'disinvested'
                    max_trades += 1

    # add one more trade if currency is held at the end
    if state == 'invested':
        max_trades += 1

    profit = (profit - 1) * 100
    max_profit = (max_profit - 1) * 100
    excess_profit = profit - buy_and_hold_profit

    return buy_and_hold_profit, profit, pred_trades, excess_profit, max_profit, max_trades


def time_series_split_testing(data, **kwargs):
    ''' Test function for the time series splits. '''
    tss = TimeSeriesSplit(**kwargs).split(data)
    for i, (train_index, test_index) in enumerate(tss):
        print(f"Fold {i}:")
        print(
            f"  Train length: {len(train_index)} | Index from {train_index[0]} to {train_index[-1]}")
        print(
            f"  Test length: {len(test_index)} | Index from {test_index[0]} to {test_index[-1]}")


def get_target_profit_per_fold(folds: int,
                               y_true: pd.Series,
                               price_data: pd.Series,
                               prediction_type: str,
                               init_state: str = 'invested',
                               **kwargs) -> tuple:
    ''' Returns profits per fold given buy-and-hold strategy and given perfect
    knowledge of the target variable.
    
    Args:
        folds (int): Number of CV folds
        y_true (pd.Series): True target variable
        price_data (pd.Series): Daily log price change, needs to be same length
            as y_true
        prediction_type (str): Either `movement` or `extrema`
        init_state (str, optional): Initial state for profit calculation. Either
            `invested` or `disinvested`. If `invested` coin is bought at time
            zero, otherwise it is only bought once a buy signal occurs.
            Defaults to `invested`.
        **kwargs: Additional keyword arguments for the function
            sklearn.model_selection.TimeSeriesSplit()
            
    Returns:
        (dict, dict): (buy-and-hold profits per fold, target profits per fold)
        
    Raises:
        Error if y_true and price_data lengths do not match
    '''
    assert len(y_true) == len(price_data)
    
    bh_profits = {}
    target_profits = {}
    
    tss = TimeSeriesSplit(n_splits=folds, **kwargs).split(y_true)
    for fold, (_, test_index) in enumerate(tss):
        
        profit = calculate_profit(price_data.iloc[test_index],
                                  y_true.iloc[test_index],
                                  y_true.iloc[test_index],
                                  prediction_type,
                                  init_state)
        
        bh_profits[fold] = profit[0]
        target_profits[fold] = profit[4]
        
    return bh_profits, target_profits


def crossvalidate_movement(model,
                           problem: str,
                           X,
                           y,
                           price_data,
                           folds: int = 5,
                           add_constant_sliding_window: bool = True,
                           constant_window_size: int = 1000,
                           cutoff_tuning: bool = False,
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
        model (sklearn.base.BaseEstimator): specified model for cross validation
        problem (str): Type of prediction problem, either `regression` or
            `classification`.
        X (pd.Dataframe): explanatory variables
        y (pd.Dataframe): target variables
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
        np.array[float]: (
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
        IndexError: If X and y do not have the same number of rows.
        TypeError: If target is not numeric.
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

    # increasing sliding window
    tss = TimeSeriesSplit(n_splits=folds, **kwargs).split(X)
    for fold, (train_index, test_index) in tqdm(enumerate(tss)):

        if skip_first and fold == 0:
            continue
        
        # predict on split
        model.fit(X.iloc[train_index], y.iloc[train_index])
        if cutoff_tuning:
            preds = model.predict_proba(X.iloc[test_index])[:, 1]
        else:
            preds = model.predict(X.iloc[test_index])
        preds = pd.Series(preds, index=X.iloc[test_index].index)

        # discretise predictions
        if problem == 'classification' and cutoff_tuning and tuning_method != 'profit':
            cutoffs = get_cutoffs(y.iloc[test_index], preds)
            if tuning_method == 'accuracy':
                cutoff = cutoffs[0]
            elif tuning_method == 'youden':
                cutoff = cutoffs[1]
            elif tuning_method == 'balanced_accuracy':
                cutoff = cutoffs[2]
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
            model.fit(X.iloc[train_index], y.iloc[train_index])
            if cutoff_tuning:
                preds = model.predict_proba(X.iloc[test_index])[:, 1]
            else:
                preds = model.predict(X.iloc[test_index])
            preds = pd.Series(preds, index=X.iloc[test_index].index)

            # discretise predictions
            if problem == 'classification' and cutoff_tuning and tuning_method != 'profit':
                cutoffs = get_cutoffs(y.iloc[test_index], preds)
                if tuning_method == 'accuracy':
                    cutoff = cutoffs[0]
                elif tuning_method == 'youden':
                    cutoff = cutoffs[1]
                elif tuning_method == 'balanced_accuracy':
                    cutoff = cutoffs[2]
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


def crossvalidate_extrema(model,
                          X_min,
                          X_max,
                          y_min,
                          y_max,
                          price_data,
                          folds: int = 5,
                          add_constant_sliding_window: bool = True,
                          constant_window_size: int = 1000,
                          cutoff_tuning: bool = False,
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
        model (sklearn.base.BaseEstimator): specified model for cross validation
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
        np.array[float]: (
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
            of rows.
        TypeError: If targets are not numeric.
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
        min_model = deepcopy(model)
        min_model = min_model.fit(X_min.iloc[train_index], y_min.iloc[train_index])
        if cutoff_tuning:
            y_pred_min = min_model.predict_proba(X_min.iloc[test_index])[:, 1]
        else:
            y_pred_min = min_model.predict(X_min.iloc[test_index])
        y_pred_min = pd.Series(y_pred_min, index=X_min.iloc[test_index].index)

        max_model = deepcopy(model)
        max_model = max_model.fit(X_max.iloc[train_index], y_max.iloc[train_index])
        if cutoff_tuning:
            y_pred_max = max_model.predict_proba(X_max.iloc[test_index])[:, 1]
        else:
            y_pred_max = max_model.predict(X_max.iloc[test_index])
        y_pred_max = pd.Series(y_pred_max, index=X_max.iloc[test_index].index)
        
        # calculate AUC ROC
        min_prediction_auc = metrics.roc_auc_score(
            y_min.iloc[test_index], y_pred_min)
        max_prediction_auc = metrics.roc_auc_score(
            y_max.iloc[test_index], y_pred_max)
        
        # discretise continuous predictions
        if cutoff_tuning and tuning_method != 'profit':
            min_cutoffs = get_cutoffs(y_min.iloc[test_index], y_pred_min)
            if tuning_method == 'accuracy':
                min_cutoff = min_cutoffs[0]
            elif tuning_method == 'youden':
                min_cutoff = min_cutoffs[1]
            elif tuning_method == 'balanced_accuracy':
                min_cutoff = min_cutoffs[2]
            y_pred_min = (y_pred_min >= min_cutoff) * 1
            max_cutoffs = get_cutoffs(y_max.iloc[test_index], y_pred_max)
            if tuning_method == 'accuracy':
                max_cutoff = max_cutoffs[0]
            elif tuning_method == 'youden':
                max_cutoff = max_cutoffs[1]
            elif tuning_method == 'balanced_accuracy':
                max_cutoff = max_cutoffs[2]
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
            min_model = deepcopy(model)
            min_model = min_model.fit(X_min.iloc[train_index], y_min.iloc[train_index])
            if cutoff_tuning:
                y_pred_min = min_model.predict_proba(X_min.iloc[test_index])[:, 1]
            else:
                y_pred_min = min_model.predict(X_min.iloc[test_index])
            y_pred_min = pd.Series(y_pred_min, index=X_min.iloc[test_index].index)

            max_model = deepcopy(model)
            max_model = max_model.fit(X_max.iloc[train_index], y_max.iloc[train_index])
            if cutoff_tuning:
                y_pred_max = max_model.predict_proba(X_max.iloc[test_index])[:, 1]
            else:
                y_pred_max = max_model.predict(X_max.iloc[test_index])
            y_pred_max = pd.Series(y_pred_max, index=X_max.iloc[test_index].index)
            
            # calculate AUC ROC
            min_prediction_auc = metrics.roc_auc_score(
                y_min.iloc[test_index], y_pred_min)
            max_prediction_auc = metrics.roc_auc_score(
                y_max.iloc[test_index], y_pred_max)
            
            # discretise continuous predictions
            if cutoff_tuning and tuning_method != 'profit':
                min_cutoffs = get_cutoffs(y_min.iloc[test_index], y_pred_min)
                if tuning_method == 'accuracy':
                    min_cutoff = min_cutoffs[0]
                elif tuning_method == 'youden':
                    min_cutoff = min_cutoffs[1]
                elif tuning_method == 'balanced_accuracy':
                    min_cutoff = min_cutoffs[2]
                y_pred_min = (y_pred_min >= min_cutoff) * 1
                max_cutoffs = get_cutoffs(y_max.iloc[test_index], y_pred_max)
                if tuning_method == 'accuracy':
                    max_cutoff = max_cutoffs[0]
                elif tuning_method == 'youden':
                    max_cutoff = max_cutoffs[1]
                elif tuning_method == 'balanced_accuracy':
                    max_cutoff = max_cutoffs[2]
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


def plot_performace(price_data,
                    y_pred,
                    prediction_type: str,
                    init_state: str = 'invested') -> plt.axis:
    ''' Plots performance of a model forecast on a given timeframe in comparison
    to the asset price.
    
    Args:
        price_data (pd.Series or pd.DataFrame): one-dimensional array of log
            price changes
        y_pred (pd.Series or pd.DataFrame): one or two-dimensional array
            of binary predictions
        prediction_type (str): `movement` or `extrema`. In the latter case
            two-dimensional predictions are expected, with column names `min`
            and `max`. 
        init_state (str, optional): Initial state for profit calculation. Either
            `invested` or `disinvested`. If `invested` coin is bought at time
            zero, otherwise it is only bought once a buy signal occurs.
            Defaults to `invested`.)
    
    Returns:
        Matplotlib axis object
    '''
    assert (price_data.index == y_pred.index).all()
    assert y_pred.isin([0, 1]).all().all()
    
    init_value = 1 # initalise value with some positive number
    
    # add asset prices for buy-and-hold baseline
    bh_value = init_value
    bh_values = []
    for i in price_data:
        bh_value *= np.exp(i)
        bh_values += [bh_value]

    # calculate profit for each day using predictions
    # initialise state with init_state (either buy or dont buy on day 1)
    state = init_state
    pred_value = init_value  
    pred_values = []
    purchases = []
    sales = []

    if prediction_type == 'movement':
        for day in price_data.index.astype(int):
            # if holding currency add price change since yesterday to profits
            if state == 'invested':
                pred_value *= np.exp(price_data[day])
            pred_values += [pred_value]
            # if predicting price increase, and not holding, buy currency
            if y_pred[day] == 1 and state == 'disinvested':
                state = 'invested'
                purchases += [1]
                sales += [0]
            # if predicting price decrease, and holding, sell currency
            elif y_pred[day] == 0 and state == 'invested':
                state = 'disinvested'
                sales += [1]
                purchases += [0]
            else:
                purchases += [0]
                sales += [0]

    if prediction_type == 'extrema':
        # check that predictions are two-dimensional
        assert y_pred.shape[1] == 2

        for day in price_data.index.astype(int):
            # if holding currency add price change since yesterday to profits
            if state == 'invested':
                pred_value *= np.exp(price_data[day])
            pred_values += [pred_value]
            # if predicting minimum, and not holding, buy currency
            if y_pred.loc[day, 'min'] == 1 and y_pred.loc[day, 'max'] == 0 and state == 'disinvested':
                state = 'invested'
                purchases += [1]
                sales += [0]
            # if predicting maximum, and holding, sell currency
            elif y_pred.loc[day, 'min'] == 0 and y_pred.loc[day, 'max'] == 1 and state == 'invested':
                state = 'disinvested'
                sales += [1]
                purchases += [0]
            else:
                purchases += [0]
                sales += [0]

    sum_trades = np.sum(purchases) + np.sum(sales)
    
    # add one more trade if currency was bought in the beginning
    if init_state=='invested':
        sum_trades += 1
    # add one more trade if currency is held at the end
    if state == 'invested':
        sum_trades += 1
   
    # plot price and model performance
    _, ax = plt.subplots()
    ax.plot(bh_values, label='Buy-an-hold portfolio', zorder=2)
    ax.plot(pred_values, label=f'Model portfolio, {sum_trades} trades', zorder=1)
    for ix in np.nonzero(np.array(purchases))[0]:
        ax.axvline(ix, color='g', alpha=.3, zorder=0)
    for ix in np.nonzero(np.array(sales))[0]:
        ax.axvline(ix, color='r', alpha=.3, zorder=0)
    # ax.set_yticks([])
    ax.set_xlabel('Days', labelpad=5)
    ax.set_ylabel('Portfolio value', labelpad=8)
    handles, _ = ax.get_legend_handles_labels()
    handles.extend([
        Line2D([0], [0], label='Asset purchases', color='g', alpha=.3),
        Line2D([0], [0], label='Asset sales', color='r', alpha=.3),
    ])
    ax.legend(handles=handles)
    
    return ax
