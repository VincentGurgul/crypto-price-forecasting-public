''' Functions for Google News data fetching. '''

import sys
sys.path.append('../../')

import re
import string
import random
import requests
import pandas as pd

from datetime import datetime, timedelta
from bs4 import BeautifulSoup

from functools import lru_cache


class GoogleNewsRSS:
    ''' Scraper for Google News RSS. '''

    def __init__(self, rss_url):

        # Fetch url
        self.response = requests.get(rss_url)

        # Parse reponse
        try:
            self.soup = BeautifulSoup(self.response.text, 'lxml')
        except Exception as e:
            print('Could not parse the xml: ', rss_url)
            print(e)

        # Extract individual elements of google news and some metadata
        self.articles = self.soup.findAll('item')
        self.size = len(self.articles)

        self.articles_dicts = [{'title': a.find('title').text,
                                'link': a.link.next_sibling.replace('\n', '').replace('\t', ''),
                                'description': a.find('description').text,
                                'pubdate': a.find('pubdate').text} for a in self.articles]

        self.urls = [d['link'] for d in self.articles_dicts if 'link' in d]
        self.titles = [d['title'] for d in self.articles_dicts if 'title' in d]
        self.descriptions = [d['description']
                             for d in self.articles_dicts if 'description' in d]
        self.publication_times = [d['pubdate']
                                  for d in self.articles_dicts if 'pubdate' in d]


@lru_cache
def convert_time(time: str):
    ''' Convert Google News date string to datetime. '''

    _format = '%d %b %Y %H:%M:%S'
    return datetime.strptime(re.sub(r'^.*?,', ',', time)[2:][:-4], _format)


def get_data(coin: str = 'BTC'):
    ''' Fetch full news data from Google News RSS feed for BTC or ETH. '''

    link = string.Template(
        'https://news.google.com/rss/search?q=CoinDesk+OR+Cointelegraph+OR+Decrypt,+$currency+OR+$symbol+after:$early_date+before:$late_date&ceid=US:en&hl=en-US&gl=US')

    if coin == 'BTC':
        currency = 'Bitcoin'
    elif coin == 'ETH':
        currency = 'Ethereum'
    else:
        raise ValueError(f'Coin not supported: {coin}')

    all_data = pd.DataFrame()

    c_date = datetime.strptime('01-10-2017', '%d-%m-%Y')

    # create iterator list for desired timeframe
    iterator = []
    while c_date <= datetime.now():
        iterator += [c_date]
        c_date += timedelta(days=1)
        
    # randomly iterate over dates to avoid detection as bot     
    random.shuffle(iterator)
    for date in iterator:

        next_date = date + timedelta(days=1)

        # Request data from Google News
        URL = link.substitute(currency=currency,
                              symbol=coin,
                              early_date=date.strftime('%Y-%m-%d'),
                              late_date=next_date.strftime('%Y-%m-%d'))
        request = GoogleNewsRSS(URL)

        response = [request.publication_times, request.titles, request.urls]
        c_data = pd.DataFrame(response).T
        c_data.columns = ['time', 'title', 'url']
        c_data['datetime'] = [convert_time(i) for i in c_data.time]
        c_data['timestamp'] = [datetime.timestamp(i) for i in c_data.datetime]
        c_data = c_data.drop(columns='time').set_index('timestamp')

        all_data = pd.concat([all_data, c_data])

    print(f'All elements fetched. ({len(iterator)}/{len(iterator)})')

    return all_data.sort_index().drop_duplicates()
