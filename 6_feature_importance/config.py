''' Script with model configurations. '''


def get_model_config(coin: str):
    ''' Returns configuration with best hyperparameters for XGBoost models. '''

    if coin == 'btc':
        model_config = {
            'n_estimators': 1280,
            'max_depth': 16,
            'learning_rate': 0.07380741943220238,
            'subsample': 0.8733533341154522,
            'colsample_bytree': 0.6159173315585706,
            'reg_alpha': 0.009370995626163203,
            'reg_lambda': 0.7375869427157563,
            'gamma': 0.1019596717471152,
            'scale_pos_weight': 0.6460127488634677
        }
    elif coin == 'eth':
        model_config = {
            'n_estimators': 227,
            'max_depth': 14,
            'learning_rate': 0.1112315929688702,
            'subsample': 0.9993037094659648,
            'colsample_bytree': 0.5354121420502694,
            'reg_alpha': 0.05788618767726967,
            'reg_lambda': 0.37078817069135234,
            'gamma': 0.6042875436264626,
            'scale_pos_weight': 0.9283572285831564
        }
                
    return model_config
