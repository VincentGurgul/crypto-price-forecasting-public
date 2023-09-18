''' Script to calculate scores for all news, tweets and reddit posts that
reflect their bullishness for the corresponsing cryptocurrency (Bitcoin or
Ethereum) using a pretrained BART-Large-MNLI model from Huggingface. '''

import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from scipy.special import softmax

import sys
sys.path.append('../../')

from utils.wrappers import timeit, telegram_notify


class BartMNLI():
    ''' Class to calculate zero-shot probabilities with the BART-Large-MNLI
    model from Huggingface. '''

    def __init__(self, softmax=True):
        ''' Initialises the BART-Large-MNLI model from the Huggingface Hub. 

        Args:
            softmax (bool, optional): If true, softmax function is applied to
                model output and the returned sentiment score is bound by
                [-1,1]. Otherwise it's bound by (-infty,infty). Defaults to True.
        '''
        MODEL = 'facebook/bart-large-mnli'

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        self.model = (AutoModelForSequenceClassification
                      .from_pretrained(MODEL)
                      .to(self.device))
        
        self.softmax = softmax

    def get_score_btc(self, text) -> float:
        ''' This function outputs a 'Bullish for Bitcoin' score for a given
        input text.
        
        Args:
            text: Input string of type `str` or `List[str]` or
                `List[List[str]]`.
            
        Returns:
            float: Value in the range (-infty,infty) or [-1,1], depending on
                softmax setting, that reflects how likely the hypothesis `This
                example is bullish for Bitcoin.` is true for the input string.
            
        Raises:
            ValueError: If input is not of type `str` or `List[str]` or
                `List[List[str]]`.
        '''
        hypothesis = 'This example is bullish for Bitcoin.'

        encoded_inputs = (self.tokenizer
                          .encode(text,
                                  hypothesis,
                                  return_tensors='pt',
                                  truncation=True,
                                  max_length=1024)
                          .to(self.device))

        output = self.model(encoded_inputs)[0][0].detach().cpu().numpy()

        if self.softmax:
            output = softmax(output)
        else:
            pass

        # The score is the difference of the positive value and the negative
        # value returned by the model
        score = output[2] - output[0]

        return score
    
    def get_score_eth(self, text) -> float:
        ''' This function outputs a 'Bullish for Ethereum' score for a given
        input text.
        
        Args:
            text: Input string of type `str` or `List[str]` or
                `List[List[str]]`.
            
        Returns:
            float: Value in the range (-infty,infty) or [-1,1], depending on
                softmax setting, that reflects how likely the hypothesis `This
                example is bullish for Ethereum.` is true for the input string.
            
        Raises:
            ValueError: If input is not of type `str` or `List[str]` or
                `List[List[str]]`.
        '''
        hypothesis = 'This example is bullish for Ethereum.'

        encoded_inputs = (self.tokenizer
                          .encode(text,
                                  hypothesis,
                                  return_tensors='pt',
                                  truncation=True,
                                  max_length=1024)
                          .to(self.device))

        output = self.model(encoded_inputs)[0][0].detach().cpu().numpy()

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

    btc_news = pd.read_parquet('../data/btc_news_processed.parquet.gzip')
    eth_news = pd.read_parquet('../data/eth_news_processed.parquet.gzip')
    reddit_r_bitcoin = pd.read_parquet('../data/reddit_r_bitcoin_processed.parquet.gzip')
    reddit_r_ethereum = pd.read_parquet('../data/reddit_r_ethereum_processed.parquet.gzip')
    btc_tweets = pd.read_parquet('../data/btc_tweets_processed.parquet.gzip')
    eth_tweets = pd.read_parquet('../data/eth_tweets_processed.parquet.gzip')
    
    model = BartMNLI(softmax=True)

    variable_name = 'bart_mnli_bullish_score'

    btc_news[variable_name] = btc_news.title.apply(model.get_score_btc)
    eth_news[variable_name] = eth_news.title.apply(model.get_score_eth)
    reddit_r_bitcoin[variable_name] = reddit_r_bitcoin.content.apply(model.get_score_btc)
    reddit_r_ethereum[variable_name] = reddit_r_ethereum.content.apply(model.get_score_eth)
    btc_tweets[variable_name] = btc_tweets.content_cleaned.apply(model.get_score_btc)
    eth_tweets[variable_name] = eth_tweets.content_cleaned.apply(model.get_score_eth)

    btc_news = btc_news.drop(columns=['title'])
    eth_news = eth_news.drop(columns=['title'])
    reddit_r_bitcoin = reddit_r_bitcoin.drop(columns=['num_comments', 'content'])
    reddit_r_ethereum = reddit_r_ethereum.drop(columns=['num_comments', 'content'])
    btc_tweets = btc_tweets.drop(columns=['content_cleaned'])
    eth_tweets = eth_tweets.drop(columns=['content_cleaned'])

    btc_news.to_parquet('btc_news_bart_mnli.parquet.gzip',
                        compression='gzip')
    eth_news.to_parquet('eth_news_bart_mnli.parquet.gzip',
                        compression='gzip')
    reddit_r_bitcoin.to_parquet('reddit_r_bitcoin_bart_mnli.parquet.gzip',
                                compression='gzip')
    reddit_r_ethereum.to_parquet('reddit_r_ethereum_bart_mnli.parquet.gzip',
                                compression='gzip')
    btc_tweets.to_parquet('btc_tweets_bart_mnli.parquet.gzip',
                          compression='gzip')
    eth_tweets.to_parquet('eth_tweets_bart_mnli.parquet.gzip',
                          compression='gzip')


if __name__=='__main__':
    
    main()
