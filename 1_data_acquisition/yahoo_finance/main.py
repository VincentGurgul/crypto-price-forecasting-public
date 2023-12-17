''' Fetch historical values for the S&P 500 index (^GSPC), its implied volatility index VIX and the COMEX gold price in USD. '''

import pandas as pd
import yfinance as yf

start_date='2011-08-06'
end_date='2023-03-31'

# Fetch historical values for the S&P 500 index
sp500_data = yf.download('^GSPC',
                         start=start_date,
                         end=end_date)

# Fetch historical values for the VIX
vix_data = yf.download('^VIX',
                       start=start_date,
                       end=end_date)

# Fetch historical COMEX gold prices in USD
gold_data = yf.download('GC=F',
                        start=start_date,
                        end=end_date)

# Concatenate into one dataframe and save
yf_data = pd.concat([sp500_data['Adj Close'],
                     sp500_data['Volume'],
                     vix_data['Adj Close'],
                     gold_data['Adj Close']], axis=1)

yf_data.columns = ['sp500_price',
                   'sp500_volume',
                   'vix',
                   'gold_usd_price']

yf_data.to_parquet('yf_data.parquet.gzip',
                   compression='gzip')
