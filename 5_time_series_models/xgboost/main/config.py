''' Script with model configurations. '''


def get_model_config(coin: str, problem: str, timeframe: int = None):
    ''' Returns configuration with best hyperparameters for XGBoost models. '''

    if coin == 'btc':
        if problem == 'regression':
            model_config = {
                'n_estimators': 170,
                'max_depth': 1,
                'learning_rate': 0.1898936404179875,
                'subsample': 0.5537194021048327,
                'colsample_bytree': 0.7647768328620687,
                'reg_alpha': 0.0018880940525556021,
                'reg_lambda': 0.01584054574727581,
                'gamma': 0.04378900204867145
            }
        elif problem == 'classification':
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
        elif problem == 'extrema':
            if timeframe == 7:
                model_config = {
                    'n_estimators': 959,
                    'max_depth': 16,
                    'learning_rate': 0.22737833093838034,
                    'subsample': 0.8736613568945655,
                    'colsample_bytree': 0.8260832320944498,
                    'reg_alpha': 0.0014763790825354992,
                    'reg_lambda': 0.005154539510291185,
                    'gamma': 0.3813086384576175,
                    'scale_pos_weight': 0.09198986011610774
                }
            elif timeframe == 14:
                model_config = {
                    'n_estimators': 934,
                    'max_depth': 15,
                    'learning_rate': 0.044264349738701046,
                    'subsample': 0.9979495775654953,
                    'colsample_bytree': 0.594328995311536,
                    'reg_alpha': 0.005430838574488611,
                    'reg_lambda': 0.03777915956497569,
                    'gamma': 0.48677148010420135,
                    'scale_pos_weight': 0.48563968832442916
                }
            elif timeframe == 21:
                model_config = {
                    'n_estimators': 429,
                    'max_depth': 16,
                    'learning_rate': 0.12269386922450651,
                    'subsample': 0.943317986745646,
                    'colsample_bytree': 0.8510305267629141,
                    'reg_alpha': 0.0016931591360216432,
                    'reg_lambda': 0.4292196418808074,
                    'gamma': 0.8391455704005655,
                    'scale_pos_weight': 0.3557214996069437
                }
    elif coin == 'eth':
        if problem == 'regression':
            model_config = {
                'n_estimators': 576,
                'max_depth': 17,
                'learning_rate': 0.16519158928444302,
                'subsample': 0.5301237949092328,
                'colsample_bytree': 0.5482695404890158,
                'reg_alpha': 0.0032181455147187797,
                'reg_lambda': 0.17913930105938314,
                'gamma': 0.10568969347802948
            }
        elif problem == 'classification':
            model_config = {
                'n_estimators': 1055,
                'max_depth': 1,
                'learning_rate': 0.023076691927919037,
                'subsample': 0.68884099976753,
                'colsample_bytree': 0.8938708344784886,
                'reg_alpha': 0.8221869321255811,
                'reg_lambda': 0.00837063851162241,
                'gamma': 0.3687237197048926,
                'scale_pos_weight': 0.7056937549242558
            }
        elif problem == 'extrema':
            if timeframe == 7:
                model_config = {
                    'n_estimators': 145,
                    'max_depth': 7,
                    'learning_rate': 0.09501170695260422,
                    'subsample': 0.6478391362276119,
                    'colsample_bytree': 0.9457284170960993,
                    'reg_alpha': 0.011452798018788487,
                    'reg_lambda': 0.5705348575601792,
                    'gamma': 0.23489896102587893,
                    'scale_pos_weight': 0.41877459905317743
                }
            if timeframe == 14:
                model_config = {
                    'n_estimators': 682,
                    'max_depth': 9,
                    'learning_rate': 0.05901916901769372,
                    'subsample': 0.9134337918650728,
                    'colsample_bytree': 0.9006143781963867,
                    'reg_alpha': 0.015186767418549976,
                    'reg_lambda': 0.11203111456920867,
                    'gamma': 0.36385277474133193,
                    'scale_pos_weight': 0.5574936413836438
                }
            if timeframe == 21:
                model_config = {
                    'n_estimators': 258,
                    'max_depth': 11,
                    'learning_rate': 0.14515698969054597,
                    'subsample': 0.7346384859145283,
                    'colsample_bytree': 0.9997089335301654,
                    'reg_alpha': 0.0015121944343383892,
                    'reg_lambda': 0.003984649872350337,
                    'gamma': 0.8549715969495703,
                    'scale_pos_weight': 0.307532886823473
                }
                
    return model_config
