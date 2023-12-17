''' Script to fetch historic news data for a cryptocurrency from the Google News RSS feed. '''

import sys
sys.path.append('../../')

from functions import get_data
from utils.telegram import sendMessage


if __name__ == '__main__':

    try:
        btc_news_data = get_data('BTC')
        btc_news_data.to_parquet(
            'btc_news_data.parquet.gzip', compression='gzip')
        sendMessage('Bitcoin news data successfully scraped \U0001F389 \U0001F389')
    except Exception as e:
        sendMessage('Error raised during bitcoin news data scraping \U0001F614')
        print('Error raised during bitcoin news data scraping: ', e)

    try:
        eth_news_data = get_data('ETH')
        eth_news_data.to_parquet(
            'eth_news_data.parquet.gzip', compression='gzip')
        sendMessage('Ethereum news data successfully scraped \U0001F389 \U0001F389')
    except Exception as e:
        sendMessage(
            'Error raised during ethereum news data scraping \U0001F614')
        print('Error raised during ethereum news data scraping: ', e)
