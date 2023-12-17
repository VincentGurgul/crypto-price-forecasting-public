''' Script to download BTC and ETH historical data from the cryptocompare API. '''

import pandas as pd
from functions import auth_key, get_data, get_balance_data

btc_id = 1182
eth_id = 7605

if __name__=='__main__':

    #--------------------------------------------------------------------------
    # Fetch Bitcoin (BTC) data
    #--------------------------------------------------------------------------

    # get (todays) price data
    feature = 'histoday'
    params = {'fsym': 'BTC',
              'tsym': 'EUR'}
    drop_columns = ['btc_price_volumefrom',
                    'btc_price_volumeto',
                    'btc_price_conversionType',
                    'btc_price_conversionSymbol']
    prefix = 'btc_price_'

    btc_price = get_data(feature, params, prefix).drop(columns=drop_columns)
    
    #--------------------------------------------------------------------------

    # get (todays) aggregated exchange volume for various currencies
    feature = 'histoday'
    currencies = ['EUR', 'USD', 'ETH']
    btc_currency_vol = pd.DataFrame()

    for i in currencies:
        params = {'fsym': 'BTC',
                  'tsym': i}
        prefix = 'btc_' + i + '_'
        drop_columns = [prefix + 'high',
                        prefix + 'low',
                        prefix + 'open',
                        prefix + 'close',
                        prefix + 'conversionType',
                        prefix + 'conversionSymbol']
        data = get_data(feature, params, prefix).drop(columns=drop_columns)
        btc_currency_vol = pd.concat([data, btc_currency_vol], axis=1)

    #--------------------------------------------------------------------------

    # get (todays) disaggregated exchange volume
    exchanges = ('Binance', 'BTSE', 'Bitci', 'Coinbase', 'Kraken', 'Bitfinex')
    btc_exchange_vol = pd.DataFrame()

    for i in exchanges:
        feature = 'exchange/symbol/histoday'
        params = {'fsym': 'BTC',
                'tsym': 'EUR',
                'e': i}
        prefix = 'btc_exchange_' + i + '_'
        data = get_data(feature, params, prefix)
        btc_exchange_vol = pd.concat([data, btc_exchange_vol], axis=1)
        
    #--------------------------------------------------------------------------

    # get (yesterdays) blockchain data
    feature = 'blockchain/histo/day'
    params = {'fsym': 'BTC',
              'auth_key': auth_key}
    drop_columns = ['btc_id', 'btc_symbol']

    btc_blockchain = (get_data(feature, params, 'btc_', itype=1)
                      .drop(columns=drop_columns))
    btc_blockchain.loc[max(btc_blockchain.index)+86400, :] = None
    btc_blockchain = btc_blockchain.shift(1)

    # --------------------------------------------------------------------------

    # get (yesterdays) balance data
    feature = 'blockchain/balancedistribution/histo/day'
    params = {'fsym': 'BTC',
              'auth_key': auth_key}
    prefix = 'btc_balance_distribution_'

    balances = get_balance_data(feature, params, prefix)
    balances.loc[max(balances.index)+86400, :] = None
    balances = balances.shift(1)

    #------------------------------------------------------------------------------

    # concatenate all dataframes into one and save
    dataframes = [btc_price,
                  btc_currency_vol,
                  btc_exchange_vol,
                  btc_blockchain,
                  balances]

    btc_data = pd.concat(dataframes, axis=1).sort_index()
    btc_data.to_parquet('btc_data.parquet.gzip', compression='gzip')


    # --------------------------------------------------------------------------
    # Fetch Ethereum (ETH) data
    # --------------------------------------------------------------------------

    # get (todays) price data
    feature = 'histoday'
    params = {'fsym': 'ETH',
              'tsym': 'EUR'}
    drop_columns = ['eth_price_volumefrom',
                    'eth_price_volumeto',
                    'eth_price_conversionType',
                    'eth_price_conversionSymbol']
    prefix = 'eth_price_'

    eth_price = get_data(feature, params, 'eth',
                         prefix).drop(columns=drop_columns)
    
    #--------------------------------------------------------------------------

    # get (todays) aggregated exchange volume for various currencies
    feature = 'histoday'
    currencies = ['EUR', 'USD', 'BTC']
    eth_currency_vol = pd.DataFrame()

    for i in currencies:
        params = {'fsym': 'ETH',
                  'tsym': i}
        prefix = 'eth_' + i + '_'
        drop_columns = [prefix + 'high',
                        prefix + 'low',
                        prefix + 'open',
                        prefix + 'close',
                        prefix + 'conversionType',
                        prefix + 'conversionSymbol']
        data = get_data(feature, params, 'eth', prefix).drop(columns=drop_columns)
        eth_currency_vol = pd.concat([data, eth_currency_vol], axis=1)

    #--------------------------------------------------------------------------

    # get (todays) disaggregated exchange volume
    exchanges = ('Binance', 'BTSE', 'Coinbase', 'Kraken', 'Bitfinex')
    eth_exchange_vol = pd.DataFrame()

    for i in exchanges:
        feature = 'exchange/symbol/histoday'
        params = {'fsym': 'ETH',
                  'tsym': 'EUR',
                  'e': i}
        prefix = 'eth_exchange_' + i + '_'
        data = get_data(feature, params, 'eth', prefix)
        eth_exchange_vol = pd.concat([data, eth_exchange_vol], axis=1)
        
    #--------------------------------------------------------------------------

    # get (yesterdays) blockchain data
    feature = 'blockchain/histo/day'
    params = {'fsym': 'ETH',
              'auth_key': auth_key}
    drop_columns = ['eth_id', 'eth_symbol']

    eth_blockchain = (get_data(feature, params, 'eth', 'eth_', itype=1)
                      .drop(columns=drop_columns))
    eth_blockchain.loc[max(eth_blockchain.index)+86400, :] = None
    eth_blockchain = eth_blockchain.shift(1)

    #------------------------------------------------------------------------------

    # get (yesterdays) staking rate
    feature = 'blockchain/staking/histoday'
    params = {'fsym': 'ETH',
              'auth_key': auth_key}
    prefix = 'eth_staking_'
    drop_columns = ['eth_staking_issued_ts', 'eth_staking_issued_date']

    eth_staking_rate = (get_data(feature, params, 'eth', prefix, itype=1)
                        .drop(columns=drop_columns))
    eth_staking_rate.loc[max(eth_staking_rate.index)+86400, :] = None
    eth_staking_rate = eth_staking_rate.shift(1)

    #------------------------------------------------------------------------------

    # concatenate all dataframes into one and save
    dataframes = [eth_price,
                  eth_currency_vol,
                  eth_exchange_vol,
                  eth_blockchain,
                  eth_staking_rate]

    eth_data = pd.concat(dataframes, axis=1).sort_index()
    eth_data.to_parquet('eth_data.parquet.gzip', compression='gzip')


    #--------------------------------------------------------------------------
    # Fetch crypto index data
    #--------------------------------------------------------------------------

    # get daily crypto indices
    index_names = ('MVDA', 'MVDALC', 'MVDAMC', 'MVDASC')
    indices = pd.DataFrame()

    for i in index_names:
        feature = 'index/histo/day'
        params = {'indexName': i,
                  'auth_key': auth_key}
        prefix = 'index_' + i + '_'
        indices = pd.concat(
            [get_data(feature, params, prefix), indices], axis=1)

    indices.sort_index().to_parquet('indices_data.parquet.gzip', compression='gzip')

    # get hourly BTC volatility data
    feature = 'index/histo/hour'
    params = {'indexName': 'BVIN',
              'auth_key': auth_key}
    prefix = 'btc_volatility_index_'

    btc_volatility = get_data(feature, params, prefix)
    btc_volatility.to_parquet(
        'btc_volatility_hourly.parquet.gzip', compression='gzip')
