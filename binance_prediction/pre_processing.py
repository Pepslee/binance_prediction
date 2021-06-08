import pandas as pd
from datetime import datetime


def load_pre_processing(df):
    df.columns = [x.strip() for x in df.columns]
    df['open_time'] = pd.to_datetime((df['open_time'] / 1000).apply(datetime.fromtimestamp))
    df = df.sort_values(by='open_time', ascending=True, ignore_index=True)
    del df['close_time']
    del df['ignore']
    df = df.rename(columns={'volume': 'volume_btc', 'number_of_trades': 'trades', 'quote_asset_volume': 'volume_usdt',
                            'taker_buy_base_asset_volume': 'taker_volume_btc',
                            'taker_buy_quote_asset_volume': 'taker_volume_usdt', 'open_time': 'date'})
    df = df.set_index('date')
    return df


def resample(df, rule):
    df_r = df.resample(rule).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'trades': 'sum',
        'volume_btc': 'sum',
        'volume_usdt': 'sum',
        'taker_volume_btc': 'sum',
        'taker_volume_usdt': 'sum'
    })
    return df_r.dropna()
