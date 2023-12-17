''' Script to fetch historic cryptocurrency data from CoinGecko. '''

from functions import get_data

if __name__ == '__main__':

    btc_data = get_data('bitcoin')
    btc_data.to_parquet('gecko_btc_data.parquet.gzip', compression='gzip')
    
    eth_data = get_data('ethereum')
    eth_data.to_parquet('gecko_eth_data.parquet.gzip', compression='gzip')
    