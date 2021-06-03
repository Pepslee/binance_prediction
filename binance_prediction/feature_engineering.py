import pandas as pandas


def feature_engineering(df):
    # GENERATE FEATURES
    df['price/volume'] = (df['high'] - df['low']) / (df['volume_btc'] + 0.0001)
    df['price/trades'] = (df['high'] - df['low']) / (df['trades'] + 0.0001)
    df['body/shadows'] = (df['close'] - df['open']).abs() / (df['high'] - df['low'] + 0.0001).abs()
    df['taker/all'] = df['taker_volume_btc'] / (df['volume_btc'] + 0.0001).abs()
    df['color'] = (df['close'] > df['open']).astype(int)
    df['two_percent_pump'] = (((df['high'] - df['open'])/df['open']) > 0.02).astype(int)
    df['two_percent_dump'] = (df['low']/df['open'] < 0.99).astype(int)
    return df
