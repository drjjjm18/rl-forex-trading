from argparse import ArgumentParser
from os import path

import pandas as pd
import yaml
import yfinance as yf

from utils import calculate_rsi


parser = ArgumentParser()
parser.add_argument('-s', '--save_path', required=False, default='data/', help='path to save data to')
parser.add_argument('-c', '--config', required=False, default='config/data_config.yaml', help='path to load data config from')

args = parser.parse_args()
save_path = args.save_path
data_config_path = args.config

try:
    with open(data_config_path, 'r') as f:
        config = yaml.safe_load(f)

    period = config['period']
    interval = config['interval']
    moving_averages = config['moving_averages']
    calculate_rsi_var = config['calculate_rsi']
    drop_close_equals_0 = config['drop_close_equals_0']
except Exception as e:
    print('failed to load config variables: ', e)

for ticker in config['tickers']:
    df = yf.download(tickers=ticker, period=period, interval=interval)
    if len(moving_averages) > 0:
        for avg in moving_averages:
            try:
                df[f'{avg}_ma'] = df['Close'].rolling(avg).mean()
            except Exception as e:
                print(f'failed to add moving average {avg}', e)

    if calculate_rsi_var:
        df['RSI'] = calculate_rsi(df)
        df['RSI'] = df['RSI'].fillna(0)

    if config['normalise_columns']:
        columns_to_normalize = df.columns
        df[columns_to_normalize] = (df[columns_to_normalize] - df[columns_to_normalize].min()) / (df[columns_to_normalize].max() - df[columns_to_normalize].min())

    if drop_close_equals_0:
        df = df[~df["Close"] == 0]

    ticker_save_path = path.join(save_path, ticker+'.csv')
    df.to_csv(ticker_save_path, index=False)