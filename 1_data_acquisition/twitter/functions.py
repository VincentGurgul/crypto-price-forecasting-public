''' Functions for Twitter API scraping. '''

import sys
sys.path.append('../../')

import pandas as pd
import snscrape.modules.twitter as sntwitter

from datetime import datetime, timedelta

from utils.wrappers import timeit, telegram_notify


class TwitterScraper():
    ''' Scrapes Bitcoin or Ethereum related tweets based on hashtags. '''

    def __init__(self,
                 coin: str,
                 min_likes: int,
                 min_retweets: int):

        self.coin = coin
        self.min_likes = min_likes
        self.min_retweets = min_retweets

        if self.coin == 'BTC':
            self.query = '#bitcoin OR #btc'
        elif self.coin == 'ETH':
            self.query = '#ethereum OR #eth'
        else:
            raise ValueError(f'Coin not supported: {self.coin}')

    def fetch_daily_tweets(self, day: str, limit: str = None):
        ''' Fetches all tweets for a given day in the format "%Y-%m-%d". '''

        all_data = pd.DataFrame()

        start_date = day
        end_date = datetime.strptime(day, '%Y-%m-%d') + timedelta(days=1)
        end_date = datetime.strftime(end_date, '%Y-%m-%d')

        query = (self.query
                 + ' min_faves:'
                 + str(self.min_likes)
                 + ' min_retweets:'
                 + str(self.min_retweets)
                 + ' lang:en since:'
                 + start_date
                 + ' until:'
                 + end_date)

        scraper = sntwitter.TwitterSearchScraper(query)

        for iteration, tweet in enumerate(scraper.get_items()):

            if limit and iteration == limit:
                break

            df = pd.DataFrame([tweet.date, tweet.date.year, tweet.date.month,
                               tweet.date.day, tweet.retweetedTweet, tweet.id,
                               tweet.hashtags, tweet.rawContent, tweet.likeCount,
                               tweet.retweetCount, tweet.replyCount,
                               tweet.user.username, tweet.user.displayname,
                               tweet.user.renderedDescription, tweet.user.verified,
                               tweet.user.favouritesCount, tweet.user.followersCount
                               ]).T

            df.columns = ['datetime', 'year', 'month', 'day', 'retweeted_tweet',
                          'tweet ID', 'hashtags', 'content', 'like_count',
                          'retweet_count', 'reply_count', 'username',
                          'user_displayname', 'user_description', 'user_verified',
                          'user_favourites_count', 'user_follower_count']

            df['timestamp'] = [date.timestamp() for date in df.datetime]
            df = df.set_index('timestamp')

            all_data = pd.concat([all_data, df])

        return all_data.sort_index()

    @timeit
    def fetch_tweets_interval(self, begin_date: str, end_date: str):
        ''' Fetches all tweets from a given begin_date in the format "%Y-%m-%d" to a given end_date in the format "%Y-%m-%d". '''

        begin_date = datetime.strptime(begin_date, '%Y-%m-%d')

        all_data = pd.DataFrame()

        fetch_date = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=1)

        counter = [0, 0]

        while fetch_date >= begin_date:
            fetch_date_str = datetime.strftime(fetch_date, '%Y-%m-%d')
            df = self.fetch_daily_tweets(fetch_date_str)

            all_data = pd.concat([all_data, df])

            fetch_date = fetch_date - timedelta(days=1)
            counter[0] += 1

            if counter[0] == 100:
                counter[1] += 1
                print(f'{counter[1]}. batch of 100 days fetched.')
                counter[0] = 0

        return all_data.sort_index()

    @telegram_notify
    def scrape_all_tweets(self, begin_date: str):
        ''' Fetches all tweets starting from a given date in the format "%Y-%m-%d". '''

        end_date = datetime.strftime(datetime.now().date(), '%Y-%m-%d')
        return self.fetch_tweets_interval(begin_date, end_date)
