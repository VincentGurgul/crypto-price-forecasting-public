''' Script to fetch historic google trends data for several keywords from the Google Trends API. '''

from functions import get_trends_data

if __name__ == '__main__':

    trends_data = get_trends_data(
        ['bitcoin',
         'ethereum',
         'cryptocurrency',
         'blockchain',
         'investing',
         ],
        timezone=str(0)  # setting timezone to UTC
    )

    trends_data.to_parquet('google_trends.parquet.gzip',
                           compression='gzip')
