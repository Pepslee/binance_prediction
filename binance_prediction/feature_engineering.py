import pandas as pandas


def feature_engineering(df):
    # GENERATE FEATURES
    df['price/volume'] = (df['high'] - df['low']) / (df['volume_btc'] + 0.0001)
    df['price/trades'] = (df['high'] - df['low']) / (df['trades'] + 0.0001)
    df['body/shadows'] = (df['close'] - df['open']).abs() / (df['high'] - df['low'] + 0.0001).abs()
    df['taker/all'] = df['taker_volume_btc'] / (df['volume_btc'] + 0.0001).abs()
    df['color'] = (df['close'] > df['open']).astype(int)
    df['(high-open)/open'] = (df['high'] - df['open'])/df['open']
    df['(open-low)/open'] = (df['open'] - df['low'])/df['open']
    df['pump_next_candle'] = (df['(open-low)/open'].shift(-1) > 0.02).astype(int)
    df['pump_three_next_candle'] = ((df['high'].rolling(3).max().shift(-3) - df['open'])/df['open'] > 0.01).astype(int)
    df['pct_change'] = df['close'].pct_change()
    df = df.dropna()
    return df
