''' Functions for numeric data preprocessing. '''

import numpy as np
import pandas as pd
import statsmodels.stats.api as sms

from termcolor import colored
from scipy.signal import argrelmin, argrelmax
from scipy.signal._peak_finding import _boolrelextrema
from arch.unitroot import ADF, PhillipsPerron, KPSS
from statsmodels.formula.api import ols


def ffill_nans(df, exclude_cols: list = [None]):
    try:
        for column in [i for i in df.columns if (i not in exclude_cols)]:
            df[column] = df[column].fillna(method='ffill')
    except:
        df = df.fillna(method='ffill')
    return df


def unit_root_testing(df: pd.DataFrame,
                      conf: float = 0.05,
                      verbose: bool = False,
                      tabsize: int = 30):
    ''' Checks stationarity of all columns of a given pandas df using
    Augmented Dickey-Fuller (ADF), Phillips-Perron (PP) and Kwiatkowski-Phillips-Schmidt-Shin (KPSS)
    unit root tests. '''
    
    if verbose:
    
        print('ADF test:\n H0: unit root, H1: stationarity\n')

        for column in df.columns:
            p_value = np.round(ADF(df[column].dropna()).pvalue, 5)
            
            # reject null hypothesis is p-value is below 5%
            if p_value <= conf:
                reject = True
            else:
                reject = False
            print(f' {column} : \tp = {p_value} {"<" if reject else ">"} {conf} \t-> {"reject" if reject else "don`t reject"} H0 @ {100*conf}% conf.'.expandtabs(tabsize))
            
        
        print('\n------------------------------------------------------------------------------------------\n')     
        print('PP test:\n H0: unit root, H1: stationarity\n')

        for column in df.columns:
            p_value = np.round(PhillipsPerron(df[column].dropna()).pvalue, 5)
            
            # reject null hypothesis is p-value is below 5%
            if p_value <= conf:
                reject = True
            else:
                reject = False
            print(f' {column} : \tp = {p_value} {"<" if reject else ">"} {conf} \t-> {"reject" if reject else "don`t reject"} H0 @ {100*conf}% conf.'.expandtabs(tabsize))
            
            
        print('\n------------------------------------------------------------------------------------------\n')     
        print('KPSS test:\n H0: stationarity, H1: unit root\n')
        for column in df.columns:
            p_value = np.round(KPSS(df[column].dropna()).pvalue, 5)
            
            # reject null hypothesis is p-value is below 5%
            if p_value <= conf:
                reject = True
            else:
                reject = False
            print(f' {column} : \tt = {p_value} {"<" if reject else ">"} {conf} \t-> {"reject" if reject else "don`t reject"} H0 @ {100*conf}% conf.'.expandtabs(tabsize))
            
    else:
        
        print('Results of ADF, PP and KPSS tests by column (p-values):\n')
        
        for column in df.columns:
            
            try:
                p_value = ADF(df[column].dropna()).pvalue
                test_ok = p_value < conf
                error = False
            except Exception as e:
                error = True
                except_name = type(e).__name__
            if error:
                ADF_result = colored(except_name, "cyan")
            elif test_ok and not error:
                ADF_result = colored(f'{p_value:.4f}', "green")
            elif not test_ok and not error:
                ADF_result = colored(f'{p_value:.4f}', "red")
                
            try:
                p_value = PhillipsPerron(df[column].dropna()).pvalue
                test_ok = p_value < conf
                error = False
            except Exception as e:
                error = True
                except_name = type(e).__name__
            if error:
                PP_result = colored(except_name, "cyan")
            elif test_ok and not error:
                PP_result = colored(f'{p_value:.4f}', "green")
            elif not test_ok and not error:
                PP_result = colored(f'{p_value:.4f}', "red")
                
            try:
                p_value = KPSS(df[column].dropna()).pvalue
                test_ok = p_value > conf
                error = False
            except Exception as e:
                error = True
                except_name = type(e).__name__
            if error:
                KPSS_result = colored(except_name, "cyan")
            elif test_ok and not error:
                KPSS_result = colored(f'{p_value:.4f}', "green")
            elif not test_ok and not error:
                KPSS_result = colored(f'{p_value:.4f}', "red")

            print(f'{column} --\t ADF: {ADF_result},\tPP: {PP_result},\tKPSS: {KPSS_result}'.expandtabs(tabsize))
            

class HeskedTesting:

    TEST_NAMES = ['White', 'Breusch-Pagan', 'Goldfeld-Quandt']

    @staticmethod
    def het_tests(series: pd.Series, test: str) -> float:
        '''
        Testing for heteroskedasticity

        :param series: Univariate time series as pd.Series
        :param test: String denoting the test. One of 'white','goldfeldquandt', or 'breuschpagan'

        :return: p-value as a float.

        If the p-value is high, we accept the null hypothesis that the data is homoskedastic
        '''
        series = series.reset_index(drop=True).reset_index()
        series.columns = ['time', 'value']
        series['time'] += 1

        olsr = ols('value ~ time', series).fit()

        if test == 'White':
            _, p_value, _, _ = sms.het_white(olsr.resid, olsr.model.exog)
        elif test == 'Goldfeld-Quandt':
            _, p_value, _ = sms.het_goldfeldquandt(
                olsr.resid, olsr.model.exog, alternative='two-sided')
        else:
            _, p_value, _, _ = sms.het_breuschpagan(
                olsr.resid, olsr.model.exog)

        return p_value

    @classmethod
    def run_all_tests(cls, df: pd.DataFrame, conf: float = 0.05, tabsize: int = 30):

        print('Results of White, Breusch-Pagan and Goldfeld-Quandt tests by column (p-values):\n')

        for column in df.columns:
            p_vals = {}
            for test in cls.TEST_NAMES:
                p_value = cls.het_tests(df[column].dropna(), test)
                if p_value <= conf:
                    p_vals[test] = colored(f'{p_value:.4f}', 'red')
                else:
                    p_vals[test] = colored(f'{p_value:.4f}', 'green')

            print(
                f'{column} --\t White: {p_vals["White"]},\tBreusch-Pagan: {p_vals["Breusch-Pagan"]},\tGoldfeld-Quandt: {p_vals["Goldfeld-Quandt"]}'.expandtabs(tabsize))
