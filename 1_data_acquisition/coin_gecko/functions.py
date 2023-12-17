''' Functions for fetching historic cryptocurrency data from CoinGecko. '''

import sys

sys.path.append('../../')

import random
import requests
import pandas as pd

from datetime import datetime, timedelta

from utils.wrappers import timeit, telegram_notify


@timeit
@telegram_notify
def get_data(coin: str = 'bitcoin', prefix: bool = True):
    ''' Get historic data from CoinGecko for BTC or ETH. '''

    url = 'https://api.coingecko.com/api/v3/coins/'

    if coin == 'bitcoin':
        start_time = '28-04-2013'
        prefix = 'btc_'
    elif coin == 'ethereum':
        start_time = '07-08-2015'
        prefix = 'eth_'
    else:
        raise ValueError(f'Coin not supported: {coin}')

    c_time = datetime.strptime(start_time, '%d-%m-%Y')

    data_final = pd.DataFrame()

    # create iterator list for desired timeframe
    iterator = []
    while c_time < datetime.now():
        iterator += [c_time]
        c_time += timedelta(days=1)
        
    # randomly iterate over dates to avoid detection as bot     
    random.shuffle(iterator)
    for date in iterator:  

        # fetch json from URL for given date
        params = {'date': datetime.strftime(date, '%d-%m-%Y')}
        response = requests.get(url + coin + '/history', params=params)
            
        # create dictionary from json
        data_dict = response.json()
            
        if list(data_dict.keys())[0] == 'error':
            raise BaseException(
                f'URL returned error: {list(data_dict.values())[0]}')
            
        # convert data json to pandas dataframe
        timestamp = date.timestamp()
        data_pandas = pd.DataFrame([{'time': timestamp}])

        try:
            for i in list(data_dict['market_data']):
                tmp = pd.DataFrame(
                    [data_dict['market_data'][i]['eur']]).rename(columns={0: i})
                data_pandas = pd.concat([data_pandas, tmp], axis=1)

            for i in list(data_dict['community_data']):
                tmp = pd.DataFrame(
                    [data_dict['community_data'][i]]).rename(columns={0: i})
                data_pandas = pd.concat([data_pandas, tmp], axis=1)

            for i in list(data_dict['developer_data']):
                tmp = pd.DataFrame(
                    [data_dict['developer_data'][i]]).rename(columns={0: i})
                data_pandas = pd.concat([data_pandas, tmp], axis=1)

            data_final = pd.concat([data_final, data_pandas])
        except:
            pass

    print(f'All elements fetched. ({len(iterator)}/{len(iterator)})')

    # set index to unix time
    data_final = data_final.set_index('time')

    # prefix each columns name with coin description
    if prefix:
        data_final = data_final.add_prefix(prefix)

    return data_final.sort_index()
