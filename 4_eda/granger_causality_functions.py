''' Functions for time series Granger causality analysis. '''

import warnings
import numpy as np
import pandas as pd

from tqdm import tqdm
from statsmodels.tsa.stattools import grangercausalitytests


def individual_granger_causality(data: pd.DataFrame,
                                 target_variable: str,
                                 test: str = 'ssr_chi2test',
                                 conf: float = 0.05):
    ''' Checks granger causality at lag 1 of all variables with respect to a
    chosen target variable. Returns dataframe with p-values of the variables
    that were included in a statistically significant regression.
    
    Args:
        data (pd.DataFrame): Dataframe with all time series, including target.
        target_variable (str): Name of target variable.
        test (str): Type of test to use - `ssr_ftest`, `ssr_chi2test`, `lrtest` or
            `params_ftest`. Defaults to `ssr_chi2test`.
        conf (float): Confidence level for the analysis. Default: 0.05, i.e.
            p-values above 5 % will not be considered.
        
    Returns:
        pd.DataFrame: List of variables with significant causal relationships
            and their corresponding p-values.
    '''
    output = pd.DataFrame()

    for var in tqdm(data.columns):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            test_result = grangercausalitytests(
                data[[target_variable, var]],
                maxlag=1,
                verbose=False,
            )
            p_value = [round(test_result[1][0][test][1], 4)]
            new_row = pd.DataFrame({'variable': var,
                                    'p_value': p_value})
            output = pd.concat([output, new_row])

    return output[output.p_value <= conf].reset_index(drop=True)


def granger_matrix(data: pd.DataFrame,
                   maxlag: int,
                   test: str = 'ssr_chi2test'):
    ''' Checks Granger causality of all possible combinations of the time series.
    The rows are the response variables, columns are the predictors. The values in the
    table are the p-values. P-values lesser than the significance level (e.g. 0.05) imply
    that the null hypothesis (coefficients of the regression are zero, i.e. X does not
    cause Y) can be rejected at that significance level.

    Args:
        data (pd.DataFrame): Input dataframe containing the time series
        maxlag (int): Amount of time lags to include in the regression
        test (str): Type of test to use - `ssr_ftest`, `ssr_chi2test`, `lrtest` or
            `params_ftest`. Defaults to `ssr_chi2test`.
            
    Returns:
        pd.DataFrame: Symmetric matrix of p-values.
    '''
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        matrix = pd.DataFrame(np.nan, columns=data.columns, index=data.columns)
        for c in tqdm(matrix.columns):
            for r in matrix.index:
                test_result = grangercausalitytests(
                    data[[r, c]],
                    maxlag=maxlag,
                    verbose=False,
                )
                p_values = [round(test_result[i+1][0][test][1], 4)
                            for i in range(maxlag)]
                min_p_value = np.min(p_values)
                matrix.loc[r, c] = min_p_value
        matrix.columns = [var + '_x' for var in data.columns]
        matrix.index = [var + '_y' for var in data.columns]

    return matrix


def get_relevant_lags(data: pd.DataFrame,
                      target_variable: str,
                      maxlag: int,
                      test: str = 'ssr_chi2test',
                      conf: float = 0.05):
    ''' Checks granger causality of all variables with respect to a chosen target
    variable. Returns dataframe with all lags that where included in a
    statistically significant regression.
    
    Args:
        data (pd.DataFrame): Dataframe with all time series, including target.
        target_variable (str): Name of target variable.
        maxlag (int): Amount of time lags to include in the regression
        test (str): Type of test to use - `ssr_ftest`, `ssr_chi2test`, `lrtest` or
            `params_ftest`. Defaults to `ssr_chi2test`.
        conf (float): Confidence level for the analysis. Default: 0.05, i.e.
            p-values above 5 % will not be considered.
        
    Returns:
        pd.DataFrame: List of variables with causal relationships and their
            relevant lags.
    '''
    output = pd.DataFrame()

    for var in tqdm(data.columns):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            test_result = grangercausalitytests(
                data[[target_variable, var]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1], 4)
                        for i in range(maxlag)]
            relevant_lags = [[x[0]
                              for x in enumerate(p_values) if x[1] <= conf]]
            new_row = pd.DataFrame({'variable': var,
                                    'relevant_lags': relevant_lags})
            output = pd.concat([output, new_row])

    return output[output['relevant_lags'].str.len() != 0].reset_index(drop=True)
