import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def feature_engineering(df):
    # GENERATE FEATURES
    new_df = pd.DataFrame()
    new_df['momentum'] = np.where(df['open'] <= df['close'], (df['high'] / df['open'] - 1), (df['low'] / df['open'] - 1))
    # df['price/volume'] = (df['high'] - df['low']) / (df['volume_btc'] + 0.0001)
    # df['price/trades'] = (df['high'] - df['low']) / (df['trades'] + 0.0001)
    # df['body/shadows'] = (df['close'] - df['open']).abs() / (df['high'] - df['low'] + 0.0001).abs()
    # df['taker/all'] = df['taker_volume_btc'] / (df['volume_btc'] + 0.0001).abs()
    # df['color'] = (df['close'] > df['open']).astype(int)
    # df['(high-open)/open'] = (df['high'] - df['open'])/df['open']
    # df['(open-low)/open'] = (df['open'] - df['low'])/df['open']
    # df['pump_next_candle'] = (df['(open-low)/open'].shift(-1) > 0.02).astype(int)
    # df['pump_three_next_candle'] = ((df['high'].rolling(3).max().shift(-3) - df['open'])/df['open'] > 0.01).astype(int)
    # df['pct_change'] = df['close'].pct_change()
    # df['momentum'].hist(bins=300, range=[-2, 2])
    # plt.show()
    new_df.index = df.index
    new_df = new_df.dropna()
    return new_df
