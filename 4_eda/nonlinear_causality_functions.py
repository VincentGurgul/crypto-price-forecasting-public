''' Functions for nonlinear time series causality analysis. '''

import numpy as np
import pandas as pd

from tqdm import tqdm
from nonlincausality import nonlincausalityLSTM


# This just takes too fucking long (1 minute per variable)
def individual_nonlinear_causality(data: pd.DataFrame,
                                   target_variable: str,
                                   conf: float = 0.05):
    ''' Checks causality using LSTM network at lag 1 of all variables with
    respect to a chosen target variable. Returns dataframe with p-values of the
    variables that were included in a statistically significant regression.
    
    Args:
        data (pd.DataFrame): Dataframe with all time series, including target.
        target_variable (str): Name of target variable.
        conf (float): Confidence level for the analysis. Default: 0.05, i.e.
            p-values above 5 % will not be considered.
        
    Returns:
        pd.DataFrame: List of variables with significant causal relationships
            and their corresponding p-values.
    '''
    split = 0.7
    data_train = data.head(int(np.round(len(data)*split)))
    data_test = data.tail(int(np.round(len(data)*(1-split))))
    output = pd.DataFrame()

    for var in tqdm(data.columns):
        test_result = nonlincausalityLSTM(
            x=data_train[[target_variable, var]].to_numpy(),
            maxlag=1,
            LSTM_layers=2,
            LSTM_neurons=[25, 25],
            Dense_layers=2,
            Dense_neurons=[100, 100],
            x_test=data_test[[target_variable, var]].to_numpy(),
            run=3,
            add_Dropout=True,
            Dropout_rate=0.01,
            epochs_num=[50, 100],
            learning_rate=[0.001, 0.0001],
            batch_size_num=128,
            verbose=False,
            plot=False,
        )
        p_value = [round(test_result[1].p_value, 4)]
        new_row = pd.DataFrame({'variable': var,
                                'p_value': p_value})
        output = pd.concat([output, new_row])

    return output[output.p_value <= conf].reset_index(drop=True)
