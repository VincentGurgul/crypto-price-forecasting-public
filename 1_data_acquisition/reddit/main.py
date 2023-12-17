''' Script to fetch all historic reddit posts from several subreddits through the Pushshift API. '''

import pytz
from datetime import datetime

from functions import get_historic_data


if __name__ == '__main__':

    before = int(datetime.now().replace(tzinfo=pytz.UTC).timestamp())

    # fetch r/Bitcoin data since 2012
    after = int(datetime(2012, 1, 1, tzinfo=pytz.UTC).timestamp())
    reddit_r_bitcoin = get_historic_data('Bitcoin', after, before)
    reddit_r_bitcoin.to_parquet('reddit_r_bitcoin.parquet.gzip',
                                compression='gzip')

    # fetch r/ethereum data since 2014
    after = int(datetime(2014, 2, 1, tzinfo=pytz.UTC).timestamp())
    reddit_r_ethereum = get_historic_data('ethereum', after, before)
    reddit_r_ethereum.to_parquet('reddit_r_ethereum.parquet.gzip',
                                 compression='gzip')
