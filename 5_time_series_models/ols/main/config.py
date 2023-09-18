''' Script with model configurations. '''


def get_model_config(coin: str, problem: str, timeframe: int = None):
    ''' Returns configuration with best hyperparameters for OLS / Logit models. '''

    if coin == 'btc':
        if problem == 'regression':
            model_config = {
                'alpha': 0.005958799012859971,
                'solver': 'lsqr',
            }
        elif problem == 'classification':
            model_config = {
                'C': 0.06782873575544636,
                'solver': 'saga',
            }
        elif problem == 'extrema':
            if timeframe == 7:
                model_config = {
                    'class_weight': {0: 1, 1: 18.211381513028112},
                    'C': 450.75978576443464,
                    'solver': 'saga',
                }
            elif timeframe == 14:
                model_config = {
                    'class_weight': {0: 1, 1: 17.022139095155925},
                    'C': 642.1143409240389,
                    'solver': 'newton-cg',
                }
            elif timeframe == 21:
                model_config = {
                    'class_weight': {0: 1, 1: 14.441339264477422},
                    'C': 0.001696346509287501,
                    'solver': 'newton-cholesky',
                }
    elif coin == 'eth':
        if problem == 'regression':
            model_config = {
                'alpha': 3.49451617307774,
                'solver': 'lsqr',
            }
        elif problem == 'classification':
            model_config = {
                'C': 0.0017810081008049813,
                'solver': 'liblinear',
            }
        elif problem == 'extrema':
            if timeframe == 7:
                model_config = {
                    'class_weight': {0: 1, 1: 2.389329198519522},
                    'C': 0.014173586510834106,
                    'solver': 'liblinear',
                }
            if timeframe == 14:
                model_config = {
                    'class_weight': {0: 1, 1: 12.308162505575911},
                    'C': 0.009517146264720696,
                    'solver': 'saga',
                }
            if timeframe == 21:
                model_config = {
                    'class_weight': {0: 1, 1: 12.083476718335945},
                    'C': 0.1595226324172412,
                    'solver': 'newton-cg',
                }
                
    return model_config
