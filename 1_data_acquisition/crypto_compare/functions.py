''' Functions for fetching CryptoCompare data. '''

import requests
import json
import pandas as pd

url = 'https://min-api.cryptocompare.com/data/'

api_key = '<API_KEY>'
auth_key = '<AUTH_KEY>'


def get_data(feature: str,
             params: dict,
             coin: str = 'btc',
             prefix: str = None,
             itype: int = 2):

    if coin=='btc':
        start_time = 1314316800
    elif coin=='eth':
        start_time = 1438819200
    else:
        raise ValueError(f'Coin not supported: {coin}')

    # fetch json from URL
    params.update({'limit': 2000})
    response = requests.get(url + feature, params)
    data_dict = json.loads(response.content)
    
    if 'Response' in data_dict and data_dict['Response']=='Error':
        raise BaseException(f'URL returned error: {data_dict["Message"]}')
    
    # convert data json to pandas dataframe
    data_pandas = pd.DataFrame()
    
    if itype==1:
        for i in data_dict['Data']['Data']:
            data_pandas = pd.concat([data_pandas, pd.DataFrame([i])])
    elif itype==2:
        for i in data_dict['Data']:
            data_pandas = pd.concat([data_pandas, pd.DataFrame([i])])
            
    # set index to unix time
    data_pandas = data_pandas.set_index('time')
    
    # drop notes if applicable
    if 'notes' in data_pandas.columns:
        data_pandas = data_pandas.drop(columns='notes')
    
    # set empty dataframe to start while loop
    data_pandas_new = (None, None)
    
    # iterate in batches of 2000 time steps until full data is received
    while data_pandas.duplicated().sum() < 1000 and data_pandas[:1].index[0] > start_time and len(data_pandas_new) > 1:
        
        # fetch json from URL   
        params.update({'toTs': data_pandas[:1].index[0]})
        response = requests.get(url + feature, params)
        data_dict = json.loads(response.content)

        # convert data json to pandas dataframe
        data_pandas_new = pd.DataFrame()

        if itype==1:
            for i in data_dict['Data']['Data']:
                data_pandas_new = pd.concat([data_pandas_new, pd.DataFrame([i])])
        elif itype==2:
            for i in data_dict['Data']:
                data_pandas_new = pd.concat([data_pandas_new, pd.DataFrame([i])])
                
        data_pandas_new = data_pandas_new.set_index('time')
        
        # merge new data into existing dataframe
        data_pandas = pd.concat([data_pandas_new[:-1], data_pandas])
        
        # drop notes if applicable
        if 'notes' in data_pandas.columns:
            data_pandas = data_pandas.drop(columns='notes')

    # prefix each columns name with feature description
    if prefix:
        data_pandas = data_pandas.add_prefix(prefix)
            
    # return only data that is newer than the oldest price data and not all zero
    data_pandas = data_pandas[data_pandas.index > start_time]
    data_pandas = data_pandas.loc[~(data_pandas==0).all(axis=1)]
    
    return data_pandas


def convert_balance_data(data_dict: dict):
    
    # convert balance data json to pandas dataframe
    data_pandas = pd.DataFrame()

    for i in range(len(data_dict['Data']['Data'])):

        x = pd.DataFrame()

        for j in data_dict['Data']['Data'][i]['balance_distribution']:
            x = pd.concat([x, pd.DataFrame([j])])

        x = pd.melt(x.drop(columns='to'), id_vars=x.columns[2:4])
        x['idx'] = x.variable + '_' + x.value.round(3).astype(str)
        x = x.drop(columns=['variable', 'value']).transpose()
        x.columns = x.iloc[-1].values
        x = x.iloc[:-1].stack().reset_index()
        x['idx'] = x.level_1 + '_' + x.level_0
        x = x.drop(columns=['level_0', 'level_1']).transpose()
        x.columns = x.iloc[-1]
        x = x.iloc[:-1]
        x['time'] = data_dict['Data']['Data'][i]['time']
        x = x.set_index('time').rename_axis(None, axis=1)
        
        data_pandas = pd.concat([data_pandas, x])   
        
    return data_pandas
    

def get_balance_data(feature: str, params: dict, prefix: str = None):

    # fetch json from URL
    params.update({'limit': 2000})
    response = requests.get(url + feature, params)
    data_dict = json.loads(response.content)
    
    if 'Response' in data_dict and data_dict['Response']=='Error':
        raise BaseException(f'URL returned error: {data_dict["Message"]}')

    # convert data json to pandas dataframe
    data_pandas = convert_balance_data(data_dict)
        
    # iterate in batches of 2000 time steps until full data is received
    while data_pandas.duplicated().sum() < 1000 and data_pandas[:1].index[0] > 1314316800:
        
        # fetch json from URL   
        params.update({'toTs': data_pandas[:1].index[0]})
        response = requests.get(url + feature, params)
        data_dict = json.loads(response.content)

        # convert data json to pandas dataframe
        data_pandas_new = convert_balance_data(data_dict)

        # merge new data into existing dataframe
        data_pandas = pd.concat([data_pandas_new[:-1], data_pandas])

    # prefix each columns name with feature description
    if prefix:
        data_pandas = data_pandas.add_prefix(prefix)
            
    # return only data that is newer than the oldest price data and not all zero
    data_pandas = data_pandas[data_pandas.index > 1314316800]
    data_pandas = data_pandas.loc[~(data_pandas==0).all(axis=1)]
    
    return data_pandas
