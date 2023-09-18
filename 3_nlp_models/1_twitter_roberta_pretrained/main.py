''' Script to calculate sentiment scores for all news, tweets and reddit posts
using a pretrained Twitter-Roberta-Base model from Huggingface. '''

import torch
import pandas as pd

from scipy.special import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import sys
sys.path.append('../../')

from utils.wrappers import timeit, telegram_notify


class TwitterRoberta():
    ''' Class to calculate sentiment scores with the Twitter-Roberta-Base model
    from Huggingface. '''

    def __init__(self, softmax=True):
        ''' Initialises latest version of the Twitter-Roberta-Base model
        from the Huggingface Hub.
        
        Args:
            softmax (bool, optional): If true, softmax function is applied to
                model output and the returned sentiment score is bound by
                [-1,1]. Otherwise it's bound by (-infty,infty). Defaults to True.
        '''
        MODEL = 'cardiffnlp/twitter-roberta-base-sentiment-latest'

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        self.model = (AutoModelForSequenceClassification
                      .from_pretrained(MODEL)
                      .to(self.device))
        
        self.softmax = softmax

    def get_score(self, text) -> float:
        ''' This function outputs a sentiment score for a given input text.
        
        Args:
            text: Input string of type `str` or `List[str]` or
                `List[List[str]]`.
            
        Returns:
            float: Value in the range (-infty,infty) or [-1,1], depending on
                softmax setting, that reflects the sentiment of the input
                string. 1 or infty := positive, -1 or -infty := negative.
            
        Raises:
            ValueError: If input is not of type `str` or `List[str]` or
                `List[List[str]]`.
        '''
        encoded_input = (self.tokenizer(text,
                                        return_tensors='pt',
                                        truncation=True,
                                        max_length=512)
                         .to(self.device))
        
        output = self.model(**encoded_input)[0][0].detach().cpu().numpy()

        if self.softmax:
            output = softmax(output)
        else:
            pass

        # The score is the difference of the positive value and the negative
        # value returned by the model
        score = output[2] - output[0]

        return score


@timeit
@telegram_notify
def main():

    # Load text data
    btc_news = pd.read_parquet(
        '../2_data_processing/text_data/btc_news_processed.parquet.gzip')
    eth_news = pd.read_parquet(
        '../2_data_processing/text_data/eth_news_processed.parquet.gzip')
    reddit_r_bitcoin = pd.read_parquet(
        '../2_data_processing/text_data/reddit_r_bitcoin_processed.parquet.gzip')
    reddit_r_ethereum = pd.read_parquet(
        '../2_data_processing/text_data/reddit_r_ethereum_processed.parquet.gzip')
    btc_tweets = pd.read_parquet(
        '../2_data_processing/text_data/btc_tweets_processed.parquet.gzip')
    eth_tweets = pd.read_parquet(
        '../2_data_processing/text_data/eth_tweets_processed.parquet.gzip')

    # Init model
    model = TwitterRoberta(softmax=True)

    # Calculate sentiment scores
    btc_news['twitter_roberta_pretrained_score'] = btc_news.title.apply(
        model.get_score)
    eth_news['twitter_roberta_pretrained_score'] = eth_news.title.apply(
        model.get_score)
    reddit_r_bitcoin['twitter_roberta_pretrained_score'] = reddit_r_bitcoin.content.apply(
        model.get_score)
    reddit_r_ethereum['twitter_roberta_pretrained_score'] = reddit_r_ethereum.content.apply(
        model.get_score)
    btc_tweets['twitter_roberta_pretrained_score'] = btc_tweets.content_cleaned.apply(
        model.get_score)
    eth_tweets['twitter_roberta_pretrained_score'] = eth_tweets.content_cleaned.apply(
        model.get_score)

    # Drop text from dataframes to save space
    btc_news = btc_news.drop(columns=['title'])
    eth_news = eth_news.drop(columns=['title'])
    reddit_r_bitcoin = reddit_r_bitcoin.drop(columns=['num_comments', 'content'])
    reddit_r_ethereum = reddit_r_ethereum.drop(columns=['num_comments', 'content'])
    btc_tweets = btc_tweets.drop(columns=['content_cleaned'])
    eth_tweets = eth_tweets.drop(columns=['content_cleaned'])

    # Save sentiment scores to parquet
    btc_news.to_parquet('btc_news_roberta_pretrained.parquet.gzip',
                        compression='gzip')
    eth_news.to_parquet('eth_news_roberta_pretrained.parquet.gzip',
                        compression='gzip')
    reddit_r_bitcoin.to_parquet('reddit_r_bitcoin_roberta_pretrained.parquet.gzip',
                                compression='gzip')
    reddit_r_ethereum.to_parquet('reddit_r_ethereum_roberta_pretrained.parquet.gzip',
                                compression='gzip')
    btc_tweets.to_parquet('btc_tweets_roberta_pretrained.parquet.gzip',
                        compression='gzip')
    eth_tweets.to_parquet('eth_tweets_roberta_pretrained.parquet.gzip',
                        compression='gzip')


if __name__=='__main__':
    
    main()
