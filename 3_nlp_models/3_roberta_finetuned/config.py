''' Script with model configurations. '''


def get_model_config(dataset):
    ''' Retuns optimal RoBERTa hyperparameter configuration for each dataset. '''

    if dataset == 'btc_news':
        model_config = {
            'learning_rate': 0.03093100743332331,
            'num_train_epochs': 8,
            'per_device_train_batch_size': 16,
            'warmup_steps': 17,
            'weight_decay': 0.15819399056330835,
        }    
    elif dataset == 'eth_news':
        model_config = {
            'learning_rate': 0.017384151867392752,
            'num_train_epochs': 8,
            'per_device_train_batch_size': 16,
            'warmup_steps': 8,
            'weight_decay': 0.0032876389951213936,
        }
    elif dataset == 'reddit_r_bitcoin':
        model_config = {
            'learning_rate': 0.007502547633181182,
            'num_train_epochs': 9,
            'per_device_train_batch_size': 64,
            'warmup_steps': 16,
            'weight_decay': 0.001071134903144872,      
        }
    elif dataset == 'reddit_r_ethereum':
        model_config = {
            'learning_rate': 0.01742353792471119,
            'num_train_epochs': 3,
            'per_device_train_batch_size': 64,
            'warmup_steps': 11,
            'weight_decay': 0.0014182807920914168,
        }
    elif dataset == 'btc_tweets':
        model_config = {
            'learning_rate': 0.004293431083541939,
            'num_train_epochs': 2,
            'per_device_train_batch_size': 32,
            'warmup_steps': 19,
            'weight_decay': 0.023605807543917796,
        }
    elif dataset == 'eth_tweets':
        model_config = {
            'learning_rate': 0.004293431083541939,
            'num_train_epochs': 2,
            'per_device_train_batch_size': 32,
            'warmup_steps': 19,
            'weight_decay': 0.023605807543917796,            
        }

    return model_config
