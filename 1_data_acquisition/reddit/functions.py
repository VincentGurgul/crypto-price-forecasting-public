''' Functions for Pushshift API scraping. '''

import sys
sys.path.append('../../')

import time
import requests
import pandas as pd

from datetime import datetime
from utils.wrappers import *


url = 'https://api.pushshift.io/reddit/search/submission'

headers = {
    'Host': 'api.pushshift.io',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'DNT': '1',
    'Alt-Used': 'api.pushshift.io',
    'Upgrade-Insecure-Requests': '1',
    'Sec-Fetch-Dest': 'document',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-Site': 'none',
    'Sec-Fetch-User': '?1',
    'Sec-GPC': '1',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/111.0',
}


@log_execution
def fetch_data(subreddit, after, before, size=1000):
    ''' Function that sends a single request to the Pushshift API. '''

    params = {
        'subreddit': subreddit,
        'after': after,
        'before': before,
        'size': size,
    }

    response = requests.get(url, headers=headers, params=params)

    while response.status_code != 200:
        time.sleep(2)
        response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        return response.json()['data']


@timeit
@telegram_notify
def get_historic_data(subreddit: str, after: int, before: int):
    ''' Fetches all historic data for a given subreddit and time frame from the Pushshift API by iteratively fetching 1000 entries. '''

    all_data = pd.DataFrame()

    while True:

        # fetch data
        posts = fetch_data(subreddit, after, before)

        if len(posts) == 0:
            break

        # transform to pandas df
        columns = ('url',
                   'utc_datetime_str',
                   'author',
                   'num_comments',
                   'score',
                   'title',
                   'selftext')
        df = pd.DataFrame(posts).loc[:, columns]

        # index by datetime
        df['time'] = [datetime.strptime(i, '%Y-%m-%d %H:%M:%S')
                      for i in df.utc_datetime_str]
        df['timestamp'] = [i.timestamp() for i in df.time]
        df = df.set_index('timestamp').sort_index()

        # concatenate
        all_data = pd.concat([all_data, df])

        # reset next time interval
        before = int(df.iloc[:1].index[0])

    return all_data.sort_index()
