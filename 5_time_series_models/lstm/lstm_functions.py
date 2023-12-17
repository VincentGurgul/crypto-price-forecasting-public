''' Functions for training an LSTM model with GPU utilisation. '''

import sys
sys.path += ['../../']

import wandb
import keras
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn import metrics
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from functions import get_cutoffs, get_max_profit_cutoff, calculate_profit


class LSTM():

    def __init__(self,
                 input_shape: tuple,
                 lstm_layer_sizes: list = [200, 150],
                 dense_layer_sizes: list = [32, 16],
                 activation: str = 'tanh',
                 dropout: float = 0.2,
                 optimizer: str = 'Adam',
                 learning_rate: float = 0.0001,
                 scaling: str = None,
                 binary_classification: bool = False,
                 seed: int = 42):
        ''' LSTM model class for binary classification or regression tasks.
        
        Args:
            input_shape (tuple): Shape of the input, should be a 2D tuple:
                (n_variables, n_timesteps)
            lstm_layer_sizes (list, optional): List of the number of neurons in
                each LSTM layer, defaults to [200, 150]. Needs to contain
                between one and three integers.
            dense_layer_sizes (list, optional): List of the number of neurons in
                each layer of the dense head, defaults to [32, 16]. Needs to
                contain between zero and three integers.
            activation (str, optional): Activation function to use in the LSTM
                layers, defaults to 'tanh'.
            dropout (float, optional): Dropout rate for the dropout layers,
                defaults to 0.2.
            optimizer (str, optional): The optimizer to use during
                training, options: 'Adam', 'RMSProp' or 'SGD', defaults to 'Adam'.
            learning_rate (float, optional): Learning rate for the optimizer,
                defaults to 0.0001.
            scaling (str, optional): Which scaler to use for feature and target
                transformation, options: [None, 'StandardScaler', 'MinMaxScaler'],
                defaults to None, i.e. no scaling.
            binary_classification (bool, optional): Whether the task is a
                binary classification problem, defaults to False.
            seed (int, optional): Seed value for numpy and tensorflow,
                defaults to 42.
                
        Raises:
            ValueError: If lstm_layer_sizes is passed an empty list.
        '''
        # Set seeds
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        # Initialise scalers
        self.scaling = scaling
        if self.scaling == 'StandardScaler':
            self.input_scaler = StandardScaler()
            self.output_scaler = StandardScaler()
        elif self.scaling == 'MinMaxScaler':
            self.input_scaler = MinMaxScaler()
            self.output_scaler = MinMaxScaler()
            
        # Set some parameters for training
        n_lstm_layers = len(lstm_layer_sizes)
        if n_lstm_layers == 0:
            raise ValueError('There needs to be at least one LSTM layer.')
        
        optimizer_map = {
            'Adam': keras.optimizers.Adam,
            'RMSprop': keras.optimizers.RMSprop,
            'SGD': keras.optimizers.SGD
        }
        optimizer_class = optimizer_map[optimizer](learning_rate)
        
        # Set loss
        if binary_classification:
            loss = 'binary_crossentropy'
        else:
            loss = 'mse'

        # Initialise model
        self.input_shape = input_shape
        self.binary_classification = binary_classification
        self.model = keras.Sequential()

        # First LSTM layer
        self.model.add(keras.layers.LSTM(lstm_layer_sizes[0],
                       activation=activation,
                       input_shape=input_shape,
                       return_sequences=True if n_lstm_layers > 1 else False))
        self.model.add(keras.layers.Dropout(dropout))

        # Additional LSTM layers
        for i in range(1, n_lstm_layers):
            self.model.add(keras.layers.LSTM(lstm_layer_sizes[i],
                                             activation=activation,
                                             return_sequences=True if i != n_lstm_layers - 1 else False))
            self.model.add(keras.layers.Dropout(dropout))

        # Dense head
        for i in range(0, len(dense_layer_sizes)):
            self.model.add(keras.layers.Dense(dense_layer_sizes[i], activation=activation))
            self.model.add(keras.layers.Dropout(dropout))

        # Final dense layer for prediction
        if binary_classification:
            self.model.add(keras.layers.Dense(1, activation='sigmoid'))
        else:
            self.model.add(keras.layers.Dense(1))

        # Compile model
        self.model.compile(optimizer=optimizer_class, loss=loss)
        
        # Check GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f'GPUs available: {len(gpus)}')
        else:
            print('No GPUs available.')
            
    def fit(self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            minority_class_weight: float = 1.,
            epochs: int = 10,
            batch_size: int = 32,
            validation_split: float = 0,
            early_stopping: bool = False,
            patience: int = 10,
            verbosity: int = 1):
        ''' Fits the LSTM model to the training data.
        
        Args:
            X_train (pd.DataFrame): Training data features, should be a
                two-dimensional multi-column dataframe with shape
                    [n_observations, n_features] where n_features = n_variables
                    * n_timesteps
            y_train (pd.Series): Training data labels.
            minority_class_weight (float): Weight assigned to the minority
                class in the case of binary classification. A larger value
                implies a greater emphasis on the positive class. Defaults to
                1.0, i.e. equal weights for both classes.
            epochs (int, optional): Number of training epochs, defaults to 10.
            batch_size (int, optional): Batch size for the training, defaults
                to 32.
            validation_split (float, optional): Proportion of training data to
                use as validation data, e.g. for early stoppping, defaults to 0,
                i.e. no validation split.
            early_stopping (bool, optional): Whether to terminate the training
                if the validation loss does not decrease below its minimum for
                as many epochs as defined by the patience parameter, defaults
                to False.
            patience (int, optional): Patience for the early stopping, defaults
                to 10.
            verbosity (int, optional): Verbosity mode for the training, defaults
                to 1.
            
        Raises:
            ValueError: If X_train does not have two column levels (for the
                variables and the timesteps).
            ValueError: If the shape of the observations in X_train is not
                equal to the `input_shape` given at class initialisation.
        '''
        if len(X_train.columns.levels) != 2:
            raise ValueError(f'Input data has {len(X_train.columns.levels)} column levels, but a multicolumn dataframe with 2 levels is expected (one with the variables and one with the lags).')
        
        if (len(X_train.columns.levels[1]), len(X_train.columns.levels[0])) != self.input_shape:
            raise ValueError(f'X_train observations contain {len(X_train.columns.levels[1])} timesteps of {len(X_train.columns.levels[0])} variables, but a shape of {self.input_shape} was expected.')
        
        # Transform labels to a np.array
        y_train = y_train.to_numpy()
        
        # Fit scaler and scale the inputs and outputs
        if self.scaling:
            X_train = X_train.copy()
            X_train[X_train.columns] = self.input_scaler.fit_transform(X_train[X_train.columns])
            y_train = self.output_scaler.fit_transform(y_train.reshape(-1, 1))

        # Reshape inputs to 3D: [samples, timesteps, features]
        X_train = X_train.fillna(0).values.reshape(X_train.shape[0], 14, int(X_train.shape[1] / 14))

        # Define callbacks
        if early_stopping:
            callback = [
                keras.callbacks.EarlyStopping(monitor='val_loss',
                                              patience=patience,
                                              min_delta=0)
            ]
        else:
            callback = []
            
        # Define class reweighing
        if self.binary_classification:
            sample_weights = np.array([
                1.0 if x == 0 else minority_class_weight if x == 1 else x for x in y_train
            ])
        else:
            sample_weights = None

        # Train model
        self.history = self.model.fit(X_train,
                                      y_train,
                                      sample_weight=sample_weights,
                                      epochs=epochs,
                                      batch_size=batch_size,
                                      verbose=verbosity,
                                      validation_split=validation_split,
                                      callbacks=callback,
                                      shuffle=False)

    def plot_loss(self):
        ''' Plots the training and validation loss of the LSTM model if
        validation_split is set during training.
        '''
        plt.plot(self.history.history['loss'], label='Training Loss')
        
        if 'val_loss' in self.history.history:
            plt.plot(self.history.history['val_loss'], label='Validation Loss')
        
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper right')
        plt.show()

    def predict_proba(self, X_test: pd.DataFrame) -> pd.Series:
        ''' Predicts continuous labels for the test data.
        
        Args:
            X_test (pd.DataFrame): Test data features, should be a
                two-dimensional multi-column dataframe with shape
                [n_observations, n_features] where n_features = n_variables
                * n_timesteps

        Returns:
            pd.Series: Predicted labels for the test data.
            
        Raises:
            ValueError: If X_test does not have two column levels (for the
                variables and the timesteps).
            ValueError: If the shape of the observations in X_test is not equal
                to the `input_shape` given at class initialisation.
        '''
        if len(X_test.columns.levels) != 2:
            raise ValueError(f'Input data has {len(X_test.columns.levels)} column levels, but a multicolumn dataframe with 2 levels is expected (one with the variables and one with the lags).')
        
        if (len(X_test.columns.levels[1]), len(X_test.columns.levels[0])) != self.input_shape:
            raise ValueError(f'X_test observations contain {len(X_test.columns.levels[1])} timesteps of {len(X_test.columns.levels[0])} variables, but a shape of {self.input_shape} was expected.')
        
        # Scale inputs
        if self.scaling:
            X_test = X_test.copy()
            X_test[X_test.columns] = self.input_scaler.transform(X_test[X_test.columns])     

        # Reshape inputs to 3D: [samples, timesteps, features]
        prediction_index = X_test.index
        X_test = X_test.fillna(0).values.reshape(X_test.shape[0], 14, int(X_test.shape[1] / 14))
        
        # Compute predictions
        predictions = self.model.predict(X_test)
        
        # Inverse transform predictions
        if self.scaling:
            predictions = self.output_scaler.inverse_transform(predictions)

        return pd.Series(predictions.reshape(-1), index=prediction_index)

    def predict(self, X_test) -> pd.Series:
        ''' Predicts the labels for the test data. If binary_classification was
        set to `True` those will be binary (discrete) predictions.
        
        Args:
            X_test (pd.DataFrame): Test data features, should be a
                two-dimensional multi-column dataframe with shape
                [n_observations, n_features] where n_features = n_variables
                * n_timesteps

        Returns:
            pd.Series: Predicted labels for the test data.
            
        Raises:
            ValueError: If X_test does not have two column levels (for the
                variables and the timesteps).
            ValueError: If the shape of the observations in X_test is not equal
                to the `input_shape` given at class initialisation.
        '''
        if self.binary_classification:
            predictions = (self.predict_proba(X_test) > 0.5) * 1
        else:
            predictions = self.predict_proba(X_test)
            
        return predictions
        
    def save(self, filepath: str):
        ''' Saves model as an HDF5 file in a specified location.
        
        Args:
            filepath (str): Path to save model to.
        '''
        self.model.save(filepath)


def crossvalidate_movement_lstm(model_args: dict,
                                training_args: dict,
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
        model_args (dict): Keyword arguments for the LSTM method __init__.
        training_args (dict): Keyword arguments for the LSTM method fit.
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
    
    input_shape = (len(X.columns.levels[1]), len(X.columns.levels[0]))

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

        print()
        
        if skip_first and fold == 0:
            continue

        # predict on split
        model = LSTM(input_shape, **model_args)
        model.fit(X.iloc[train_index], y.iloc[train_index], **training_args)
        if cutoff_tuning:
            preds = model.predict_proba(X.iloc[test_index])
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
        try:
            prediction_auc = metrics.roc_auc_score(
                y_discrete.iloc[test_index], preds)
            prediction_acc = metrics.accuracy_score(
                y_discrete.iloc[test_index], preds)
        except:
            prediction_auc = np.nan
            prediction_acc = np.nan

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
            
        print()

    # constant sliding window
    if add_constant_sliding_window:
        tss = TimeSeriesSplit(
            n_splits=folds, max_train_size=constant_window_size, **kwargs).split(X)
        for fold, (train_index, test_index) in tqdm(enumerate(tss)):
            
            if skip_first and fold == 0:
                continue

            # predict on split
            model = LSTM(input_shape, **model_args)
            model.fit(X.iloc[train_index], y.iloc[train_index], **training_args)
            if cutoff_tuning:
                preds = model.predict_proba(X.iloc[test_index])
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
            try:
                prediction_auc = metrics.roc_auc_score(
                    y_discrete.iloc[test_index], preds)
                prediction_acc = metrics.accuracy_score(
                    y_discrete.iloc[test_index], preds)
            except:
                prediction_auc = np.nan
                prediction_acc = np.nan

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


def crossvalidate_extrema_lstm(model_args: dict,
                               training_args: dict,
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
        model_args (dict): Keyword arguments for the LSTM method __init__.
        training_args (dict): Keyword arguments for the LSTM method fit.
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
    
    input_shape_min = (len(X_min.columns.levels[1]), len(X_min.columns.levels[0]))
    input_shape_max = (len(X_max.columns.levels[1]), len(X_max.columns.levels[0]))
    

    results = []
    
    if skip_first:
        folds += 1

    # increasing sliding window
    tss = TimeSeriesSplit(n_splits=folds, **kwargs).split(X_min)
    for fold, (train_index, test_index) in tqdm(enumerate(tss)):
        
        print()
        
        if skip_first and fold == 0:
            continue

        # fit models and predict
        min_model = LSTM(input_shape_min, **model_args)
        min_model.fit(X_min.iloc[train_index],
                      y_min.iloc[train_index],
                      **training_args)
        if cutoff_tuning:
            y_pred_min = min_model.predict_proba(X_min.iloc[test_index])
        else:
            y_pred_min = min_model.predict(X_min.iloc[test_index])
        y_pred_min = pd.Series(y_pred_min, index=X_min.iloc[test_index].index)

        max_model = LSTM(input_shape_max, **model_args)
        max_model.fit(X_max.iloc[train_index],
                      y_max.iloc[train_index],
                      **training_args)
        if cutoff_tuning:
            y_pred_max = max_model.predict_proba(X_max.iloc[test_index])
        else:
            y_pred_max = max_model.predict(X_max.iloc[test_index])
        y_pred_max = pd.Series(y_pred_max, index=X_max.iloc[test_index].index)
        
        # calculate AUC ROC
        try:
            min_prediction_auc = metrics.roc_auc_score(
                y_min.iloc[test_index], y_pred_min)
            max_prediction_auc = metrics.roc_auc_score(
                y_max.iloc[test_index], y_pred_max)
        except:
            min_prediction_auc = np.nan
            max_prediction_auc = np.nan
        
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
        try:
            min_prediction_acc = metrics.accuracy_score(
                y_min.iloc[test_index], y_pred_min)
            max_prediction_acc = metrics.accuracy_score(
                y_max.iloc[test_index], y_pred_max)
        except:
            min_prediction_acc = np.nan
            max_prediction_acc = np.nan

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
            
        print()

    # constant sliding window
    if add_constant_sliding_window:
        tss = TimeSeriesSplit(
            n_splits=folds, max_train_size=constant_window_size, **kwargs).split(X_min)
        for fold, (train_index, test_index) in tqdm(enumerate(tss)):
            
            if skip_first and fold == 0:
                continue

            # fit models and predict
            min_model = LSTM(input_shape_min, **model_args)
            min_model.fit(X_min.iloc[train_index],
                          y_min.iloc[train_index],
                          **training_args)
            if cutoff_tuning:
                y_pred_min = min_model.predict_proba(X_min.iloc[test_index])
            else:
                y_pred_min = min_model.predict(X_min.iloc[test_index])
            y_pred_min = pd.Series(y_pred_min, index=X_min.iloc[test_index].index)

            max_model = LSTM(input_shape_max, **model_args)
            max_model.fit(X_max.iloc[train_index],
                          y_max.iloc[train_index],
                          **training_args)
            if cutoff_tuning:
                y_pred_max = max_model.predict_proba(X_max.iloc[test_index])
            else:
                y_pred_max = max_model.predict(X_max.iloc[test_index])
            y_pred_max = pd.Series(y_pred_max, index=X_max.iloc[test_index].index)
            
            # calculate AUC ROC
            try:
                min_prediction_auc = metrics.roc_auc_score(
                    y_min.iloc[test_index], y_pred_min)
                max_prediction_auc = metrics.roc_auc_score(
                    y_max.iloc[test_index], y_pred_max)
            except:
                min_prediction_auc = np.nan
                max_prediction_auc = np.nan
            
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
            try:
                min_prediction_acc = metrics.accuracy_score(
                    y_min.iloc[test_index], y_pred_min)
                max_prediction_acc = metrics.accuracy_score(
                    y_max.iloc[test_index], y_pred_max)
            except:
                min_prediction_acc = np.nan
                max_prediction_acc = np.nan

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
