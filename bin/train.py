#!/usr/bin/env python


from datetime import date
import os
import numpy as np
import pandas as pd

from binance_prediction.data_generator import train_test_split, TrainDataGenerator, TestDataGenerator, split_candles_dataset, DataGenerator
from binance_prediction.feature_engineering import feature_engineering
from binance_prediction.pre_processing import load_pre_processing, resample
from binance_prediction.models import build_model
from binance_prediction.callbacks import get_callbacks
from binance_prediction.loss import get_loss
from binance_prediction import diagram

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = os.path.join(project_dir, 'XRPUSDT_17.08.2017_08.06.2021.csv')

df = pd.read_csv(path)

df = load_pre_processing(df)

df = resample(df, '1Min')

# diagram.draw_candles(df)

# FILTER BY 2018+
# df = df[df.index > pd.Timestamp(date(2018, 1, 1))]

df = feature_engineering(df)

# TRAIN/TEST SPLIT
WIN = 12
BATCH_SIZE = 100
train_idx, test_idx, new_df, timestamps = split_candles_dataset(df, window_size=WIN)
# X_train, Y_train, X_test, Y_test = train_test_split(df)

test_data_set = DataGenerator(new_df, test_idx, WIN, batch_size=BATCH_SIZE)
train_data_set = DataGenerator(new_df, train_idx, WIN, batch_size=BATCH_SIZE)

# for i in test_data_set:
#     print(i[0], i[1])
#     print('--')

# train_batch_size = 100
# test_batch_size = 100

# train_data_set = TrainDataGenerator(X_train, Y_train, batch_size=train_batch_size, win=WIN)
# test_data_set = TestDataGenerator(X_test, Y_test, batch_size=test_batch_size, win=WIN)
#
# print('TRAIN: count of pump candles', Y_train.sum(), 'common count', Y_train.shape[0])
# print('TEST: count of pump candles', Y_test.sum(), 'common count', Y_test.shape[0])
#
#
model = build_model((WIN, new_df.shape[1]), get_loss())
print(model.summary())
#
model.fit(train_data_set, steps_per_epoch=len(train_data_set), epochs=5000,
          validation_data=test_data_set, validation_steps=len(test_data_set), verbose=1,
          callbacks=get_callbacks()
          )
#
# x, y = next(train_data_set)
# res = model(x.astype(np.float32)).numpy()
# for r, t in zip(res, y):
#     print(r, t)
