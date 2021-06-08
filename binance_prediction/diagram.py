import mplfinance as fplt


def draw_candles(data):
    max_date = data.index.max()
    min_date = data.index.min()
    fplt.plot(
        data,
        type='candle',
        style='charles',
        title=f'{min_date} - {max_date}',
        ylabel='Price'
    )
