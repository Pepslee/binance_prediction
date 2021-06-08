import numpy as np
import random
import pandas as pd
from datetime import date
from random import shuffle


def split_candles_dataset(df, window_size: int, test_percentage: int = 10, method: str = 'random'):
    if 0 > test_percentage > 100:
        raise ValueError(f'The percentage is either more than 100 or less than 0: {test_percentage}')
    # Randomly sample {test_percentage}% of your dataframe
    test_indexes = df[:-window_size].sample(frac=test_percentage / 100, axis='index').index.tolist()
    train_indexes = df[~df.index.isin(test_indexes)][:-window_size].index.tolist()
    return train_indexes, test_indexes


def train_test_split(df):
    df_train = df[df.index < pd.Timestamp(date(2021, 1, 1))]
    df_test = df[df.index >= pd.Timestamp(date(2021, 1, 1))]
    Y_train = df_train['pump_three_next_candle']
    X_train = df_train.drop(['pump_next_candle', 'pump_three_next_candle'], axis=1)

    Y_test = df_test['pump_three_next_candle']
    X_test = df_test.drop(['pump_next_candle', 'pump_three_next_candle'], axis=1)
    return X_train, Y_train, X_test, Y_test


class DataGenerator:
    def __init__(self, df, indices, batch_size=10, is_shuffle=True):
        self.indices = indices
        self.batch_size = batch_size
        self.df = df

        if is_shuffle:
            shuffle(self.indices)

    def __len__(self):
        return int(len(self.indices) / self.batch_size)

    def __iter__(self):
        for i, index in enumerate(self.indices[::self.batch_size]):
            # so as not to go beyond the list of indices in the second loop
            if i * self.batch_size + self.batch_size > len(self.indices):
                break
            batch = [self.df.loc[self.indices[i + r]] for r in range(self.batch_size)]
            yield batch



class TrainDataGenerator:
    def __init__(self, df_x, df_y, batch_size=10, win=60):
        self.X = df_x.to_numpy()
        self.Y = df_y.to_numpy()
        self.win = win
        self.batch_size = batch_size
        self.length = self.Y.shape[0]
        self.indexes = np.array(list(range(win, self.length)))
        self.positive = self.Y[self.indexes] == 1
        self.negative = self.Y[self.indexes] == 0
        self.positive_indexes = self.indexes[self.positive]
        self.negative_indexes = self.indexes[self.negative]
        random.shuffle(self.positive_indexes)
        random.shuffle(self.negative_indexes)
        self.positive_position = 0
        self.negative_position = 0
        self.on_epoch_end()

    def __next__(self):
        # positive indexes
        if self.positive_position >= len(self.positive_indexes) - int(self.batch_size/2):
            self.positive_position = 0
            random.shuffle(self.positive_indexes)
        pos_ind = self.positive_indexes[self.positive_position:self.positive_position + int(self.batch_size/2)]

        # negative indexes
        if self.negative_position >= len(self.negative_indexes) - int(self.batch_size/2):
            self.negative_position = 0
            random.shuffle(self.negative_indexes)
        neg_ind = self.negative_indexes[self.negative_position:self.negative_position + int(self.batch_size/2)]

        ind = np.append(pos_ind, neg_ind)

        batch_x = list()
        batch_y = list()
        # TODO: разбить датасет начасти по классам для балансировки в батче
        for i in ind:
            sample_x = self.X[i - self.win + 1:i + 1]
            sample_y = self.Y[i]
            batch_x.append(np.expand_dims(sample_x, axis=-1))
            batch_y.append([sample_y])

        self.positive_position += int(self.batch_size/2)
        self.negative_position += int(self.batch_size/2)
        return np.array(batch_x).astype(np.float32), np.array(batch_y).astype(np.float32)

    def __iter__(self):
        while (True):
            yield self.__next__()

    def __len__(self):
        return int(self.length / self.batch_size)

    def on_epoch_end(self):
        random.shuffle(self.indexes)


class TestDataGenerator:
    def __init__(self, df_x, df_y, batch_size=10, win=60):
        self.X = df_x.to_numpy()
        self.Y = df_y.to_numpy()
        self.win = win
        self.batch_size = batch_size
        self.length = self.Y.shape[0]
        self.indexes = list(range(win, self.length))
        self.pos = 0
        self.on_epoch_end()

    def __next__(self):
        if self.pos >= len(self.indexes) - self.batch_size:
            self.pos = 0
            self.on_epoch_end()
        ind = self.indexes[self.pos:self.pos + self.batch_size]
        batch_x = list()
        batch_y = list()
        # TODO: разбить датасет начасти по классам для балансировки в батче
        
        for i in ind:
            sample_x = self.X[i - self.win + 1:i+1]
            sample_y = self.Y[i]
            batch_x.append(np.expand_dims(sample_x, axis=-1))
            batch_y.append([sample_y])

        self.pos += self.batch_size
        return np.array(batch_x).astype(np.float32), np.array(batch_y).astype(np.float32)

    def __iter__(self):
        while (True):
            yield self.__next__()

    def __len__(self):
        return int(self.length / self.batch_size)

    def on_epoch_end(self):
        ...
