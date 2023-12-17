''' Functions for Google Trends API scraping. '''

import sys
sys.path.append('../../')

import time
import json
import urllib
import requests
import pandas as pd

from io import StringIO
from datetime import datetime, timedelta

from config import *
from utils.wrappers import *


@log_execution
def fetch_explore_api(hl, timezone, keyword, from_date, to_date):
    ''' Fetches data required for a call to the widgets API from the explore API. '''
    
    query = urllib.parse.urlencode(
        [
            ("hl", hl),
            ("tz", timezone),
            ("req", json.dumps(
                {
                    "comparisonItem": [{"keyword": str(keyword),
                                        "geo": "",
                                        "time": datetime.strftime(from_date, '%Y-%m-%d') + ' ' + datetime.strftime(to_date, '%Y-%m-%d')}],
                    "category": 0,
                    "property": ""
                }
            ))
        ]
    )
    
    response = requests.get(explore_url + '?' + query,
                           cookies=cookies,
                           headers=headers)
    
    while response.status_code == 429:
        time.sleep(2)
        response = requests.get(explore_url + '?' + query,
                                cookies=cookies,
                                headers=headers)
        
    if response.status_code != 200:
        raise Exception(f'Explore API returned error. Status code: {response.status_code}')
    
    return json.loads(response.text[5:])['widgets']

    
@log_execution
def fetch_widget_api(params: dict):
    ''' Fetches Google Trends data from the widgets API. '''

    response = requests.get(
        widget_url,
        params=params,
        cookies=cookies,
        headers=headers,
    )
    
    return response


@telegram_notify
def get_trends_data(keywords: list,
                    begin_date: datetime = None,
                    timezone: str = str(-60),
                    hl: str = 'en'):
    ''' Downloads entire Google Trends data for a given keyword and starting date. '''

    if begin_date == None:
        begin_date = datetime.strptime('2011-08-26', '%Y-%m-%d').date()
        
    end_date = datetime.now().date()
    
    all_data = pd.DataFrame()

    # iterate over all keywords
    for keyword in keywords:
        
        to_date = end_date
        from_date = end_date - timedelta(days=250)

        complete_kw_data = pd.DataFrame()

        # iteratively fetch trends data 250 days at a time
        while True:

            # fetch token and request info from explore api
            widgets = fetch_explore_api(hl, timezone, keyword, from_date, to_date)
            token = widgets[0]['token']
            request = widgets[0]['request']

            params = {
                'req': json.dumps(request),
                'token': token,
                'tz': timezone,
            }
            
            # fetch data csv from widget api
            response = fetch_widget_api(params)

            # convert csv string to pandas df
            df = pd.read_csv(StringIO(response.text))
            df.columns = [keyword]
            
            # append 250 days of data to keyword dataframe
            complete_kw_data = pd.concat([complete_kw_data, df.iloc[1:]])
            
            # reset dates by 250 days, or set to begin date, or end loop
            to_date = from_date
            if from_date - timedelta(days=250) >= begin_date:
                from_date = from_date - timedelta(days=250)
            elif from_date - timedelta(days=250) < begin_date and from_date > begin_date:
                from_date = begin_date
            else:
                break
    
        # append keyword data to final dataframe
        all_data = pd.concat([all_data, complete_kw_data], axis=1)

    # return dataframe with all keywords sorted by time
    return all_data.sort_index()
