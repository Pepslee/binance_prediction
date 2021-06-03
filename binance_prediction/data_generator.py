import numpy as np
import random


class DataGenerator:
    def __init__(self, df, batch_size=10, win=60, future_win=3, phase='train'):
        self.df = df
        self.win = win
        self.batch_size = batch_size
        self.phase = phase
        self.length = len(self.df)
        self.future_win = future_win
        self.indexes = list(range(self.length - win - self.future_win))
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
            sample_x = self.df.iloc[i:i + self.win].to_numpy()
            pump = (self.df.iloc[i + self.win+self.future_win]['high'] - self.df.iloc[i + self.win]['open'])/self.df.iloc[i + self.win]['open'] > 0.02
            normal = 0 if pump == 1 else 1
            sample_y = np.array([normal, pump])
            batch_x.append(np.expand_dims(sample_x, axis=-1))
            # sample_y = tf.keras.utils.to_categorical(
            #     5, num_classes=2, dtype='float32'
            # )
            # batch_y.append(sample_y)
            batch_y.append([pump])

        self.pos += self.batch_size
        return np.array(batch_x).astype(np.float32), np.array(batch_y).astype(np.float32)

    def __iter__(self):
        while (True):
            yield self.__next__()

    def __len__(self):
        return int(self.length / self.batch_size)

    def on_epoch_end(self):
        if self.phase == 'train':
            random.shuffle(self.indexes)
