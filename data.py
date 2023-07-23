import yfinance as yf
from argparse import ArgumentParser
import pandas as pd

def calculate_rsi(data, window=14):
    close_prices = data['Close']
    delta = close_prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

parser = ArgumentParser()
parser.add_argument('-t', '--tickers', required=True, help='stock to get data for')
parser.add_argument('-i', '--interval', required=False, default='60m', help='data interval')
parser.add_argument('-p', '--period', required=False, default='730d', help='period to get data for')
parser.add_argument('-s', '--save_path', required=True, help='path to save data to')

args = parser.parse_args()
tickers = args.tickers
interval = args.interval
period = args.period
save_path = args.save_path

df = yf.download(tickers=tickers, period=period, interval=interval)
df['1_day_ma'] = df['Close'].rolling('1D').mean()
df['3_day_ma'] = df['Close'].rolling('3D').mean()
df['7_day_ma'] = df['Close'].rolling('7D').mean()
df['RSI'] = calculate_rsi(df)
df['RSI'] = df['RSI'].fillna(0)

columns_to_normalize = df.columns
df[columns_to_normalize] = (df[columns_to_normalize] - df[columns_to_normalize].min()) / (df[columns_to_normalize].max() - df[columns_to_normalize].min())

df.to_csv(save_path, index=False)