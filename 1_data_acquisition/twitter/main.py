''' Script to fetch historic tweets through the Twitter API. '''

from functions import TwitterScraper


if __name__=='__main__':
    
    # Scrape Bitcoin tweets
    BitcoinScraper = TwitterScraper(coin='BTC',
                                    min_likes=8,
                                    min_retweets=3)

    btc_tweets = BitcoinScraper.scrape_all_tweets('2014-01-01')

    btc_tweets.to_parquet('btc_tweets.parquet.gzip',
                          compression='gzip')

    # Scrape Ethereum tweets
    EthereumScraper = TwitterScraper(coin='ETH',
                                     min_likes=5,
                                     min_retweets=2)

    eth_tweets = EthereumScraper.scrape_all_tweets('2015-08-06')

    eth_tweets.to_parquet('eth_tweets.parquet.gzip',
                          compression='gzip')
