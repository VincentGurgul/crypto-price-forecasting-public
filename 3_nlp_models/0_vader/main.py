''' Script to calculate sentiment scores for all news, tweets and reddit posts
using the open source sentiment dictionary VADER. '''

import pandas as pd

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import sys
sys.path.append('../../')

from utils.wrappers import timeit, telegram_notify


def get_score(text: str) -> float:
    ''' This function outputs a sentiment score for a given input text.
    
    Args:
        text (str): Input string of text.
        
    Returns:
        float: Value in the range [-1,1], that reflects the sentiment of the
            input string. 1 := positive, -1 := negative.
        
    Raises:
        ValueError: If input is not of type `str`.
    '''
    
    if type(text) != str:
        raise ValueError(f'Invalid input type `{type(text).__name__}`. Please provide an input of type `str`.')

    positive_score = SentimentIntensityAnalyzer().polarity_scores(text)['pos']
    negative_score = SentimentIntensityAnalyzer().polarity_scores(text)['neg']
    
    # The score is the difference of the positive value and the negative
    # value returned by the model
    return positive_score - negative_score


@timeit
@telegram_notify
def main():

    # Load text data
    btc_news = pd.read_parquet(
        '../../2_data_processing/text_data/btc_news_processed.parquet.gzip')
    eth_news = pd.read_parquet(
        '../../2_data_processing/text_data/eth_news_processed.parquet.gzip')
    reddit_r_bitcoin = pd.read_parquet(
        '../../2_data_processing/text_data/reddit_r_bitcoin_processed.parquet.gzip')
    reddit_r_ethereum = pd.read_parquet(
        '../../2_data_processing/text_data/reddit_r_ethereum_processed.parquet.gzip')
    btc_tweets = pd.read_parquet(
        '../../2_data_processing/text_data/btc_tweets_processed.parquet.gzip')
    eth_tweets = pd.read_parquet(
        '../../2_data_processing/text_data/eth_tweets_processed.parquet.gzip')

    # Calculate sentiment scores
    btc_news['vader_score'] = btc_news.title.apply(get_score)
    eth_news['vader_score'] = eth_news.title.apply(get_score)
    reddit_r_bitcoin['vader_score'] = reddit_r_bitcoin.content.apply(get_score)
    reddit_r_ethereum['vader_score'] = reddit_r_ethereum.content.apply(get_score)
    btc_tweets['vader_score'] = btc_tweets.content_cleaned.apply(get_score)
    eth_tweets['vader_score'] = eth_tweets.content_cleaned.apply(get_score)

    # Drop text from dataframes to save space
    btc_news = btc_news.drop(columns=['title'])
    eth_news = eth_news.drop(columns=['title'])
    reddit_r_bitcoin = reddit_r_bitcoin.drop(columns=['num_comments', 'content'])
    reddit_r_ethereum = reddit_r_ethereum.drop(columns=['num_comments', 'content'])
    btc_tweets = btc_tweets.drop(columns=['content_cleaned'])
    eth_tweets = eth_tweets.drop(columns=['content_cleaned'])

    # Save sentiment scores to parquet
    btc_news.to_parquet('btc_news_vader.parquet.gzip',
                        compression='gzip')
    eth_news.to_parquet('eth_news_vader.parquet.gzip',
                        compression='gzip')
    reddit_r_bitcoin.to_parquet('reddit_r_bitcoin_vader.parquet.gzip',
                                compression='gzip')
    reddit_r_ethereum.to_parquet('reddit_r_ethereum_vader.parquet.gzip',
                                compression='gzip')
    btc_tweets.to_parquet('btc_tweets_vader.parquet.gzip',
                        compression='gzip')
    eth_tweets.to_parquet('eth_tweets_vader.parquet.gzip',
                        compression='gzip')


if __name__=='__main__':
    
    main()
