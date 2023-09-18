''' Script with model configurations. '''


def get_model_config(coin: str, problem: str, timeframe: int = None):
    ''' Returns configuration with best hyperparameters for MLP models. '''

    if coin == 'btc':
        if problem == 'regression':
            model_config = {
                'hidden_layer_sizes': (107, 114),
                'activation': 'logistic',
                'solver': 'sgd',
                'alpha': 0.000764934410726945,
                'learning_rate_init': 0.02064349176145803,
                'batch_size': 32
            }
        elif problem == 'classification':
            model_config = {
                'hidden_layer_sizes': (66, 188),
                'activation': 'logistic',
                'solver': 'sgd',
                'alpha': 0.005016935006504581,
                'learning_rate_init': 0.08233056026812441,
                'batch_size': 16
            }
        elif problem == 'extrema':
            if timeframe == 7:
                model_config = {
                    'hidden_layer_sizes': (132),
                    'activation': 'logistic',
                    'solver': 'adam',
                    'alpha': 0.0011893872810639923,
                    'learning_rate_init': 0.0020473243977996406,
                    'batch_size': 128
                }
            elif timeframe == 14:
                model_config = {
                    'hidden_layer_sizes': (11, 100, 89, 101),
                    'activation': 'relu',
                    'solver': 'sgd',
                    'alpha': 0.06573589214159975,
                    'learning_rate_init': 0.0012754982045368052,
                    'batch_size': 128
                }
            elif timeframe == 21:
                model_config = {
                    'hidden_layer_sizes': (67, 29, 143, 89),
                    'activation': 'logistic',
                    'solver': 'adam',
                    'alpha': 0.00011496871398844123,
                    'learning_rate_init': 0.014283793775515849,
                    'batch_size': 64
                }
    elif coin == 'eth':
        if problem == 'regression':
            model_config = {
                'hidden_layer_sizes': (98),
                'activation': 'identity',
                'solver': 'adam',
                'alpha': 0.07341448451784112,
                'learning_rate_init': 0.0766784049909947,
                'batch_size': 64
            }
        elif problem == 'classification':
            model_config = {
                'hidden_layer_sizes': (73),
                'activation': 'logistic',
                'solver': 'sgd',
                'alpha': 0.00020696016516304964,
                'learning_rate_init': 0.001398939785239579,
                'batch_size': 64
            }
        elif problem == 'extrema':
            if timeframe == 7:
                model_config = {
                    'hidden_layer_sizes': (77),
                    'activation': 'relu',
                    'solver': 'sgd',
                    'alpha': 0.03973583831402757,
                    'learning_rate_init': 0.02389414020248098,
                    'batch_size': 32
                }
            if timeframe == 14:
                model_config = {
                    'hidden_layer_sizes': (175),
                    'activation': 'logistic',
                    'solver': 'lbfgs',
                    'alpha': 0.013439707080241389,
                    'learning_rate_init': 0.015607755568889034,
                    'batch_size': 32
                }
            if timeframe == 21:
                model_config = {
                    'hidden_layer_sizes': (161, 181, 29),
                    'activation': 'relu',
                    'solver': 'lbfgs',
                    'alpha': 0.022833370006935458,
                    'learning_rate_init': 0.001901927842041302,
                    'batch_size': 16
                }
                
    return model_config
