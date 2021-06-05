#!/usr/bin/env python


from datetime import date
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa

from binance_prediction.callbacks import Metrics
from binance_prediction.data_generator import train_test_split, TrainDataGenerator, TestDataGenerator
from binance_prediction.feature_engineering import feature_engineering
from binance_prediction.pre_processing import load_pre_processing, resample

print(tf.__version__)

regularizer = tf.keras.regularizers.l2(0.01)

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = os.path.join(project_dir, 'BTCUSDT_2017-08-17_2021-05-21.csv')

df = pd.read_csv(path)
df = load_pre_processing(df)

df = resample(df, '60Min')

# FILTER BY 2018+
df = df[df.index > pd.Timestamp(date(2018, 1, 1))]

df = feature_engineering(df)

# TRAIN/TEST SPLIT
X_train, Y_train, X_test, Y_test = train_test_split(df)

WIN = 240
train_batch_size = 100
test_batch_size = 100

train_data_set = TrainDataGenerator(X_train, Y_train, batch_size=train_batch_size, win=WIN)
test_data_set = TestDataGenerator(X_test, Y_test, batch_size=test_batch_size, win=WIN)

# print(f'train length: {len(df_train)}', f'batch_size: {train_batch_size}',
#       f'samples:{len(df_train) / train_batch_size}')
# print(f'test length: {len(df_test)}', f'batch_size: {test_batch_size}', f'samples:{len(df_test) / test_batch_size}')
#
print('TRAIN: count of pump candles', Y_train.sum(), 'common count', Y_train.shape[0])
print('TEST: count of pump candles', Y_test.sum(), 'common count', Y_test.shape[0])
#

model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(WIN, X_train.shape[1], 1)),
    tf.keras.layers.Conv2D(4, (5, 3), strides=(1, 1), kernel_regularizer=regularizer, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(4, (5, 3), strides=(1, 1), kernel_regularizer=regularizer, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(12, (5, 3), strides=(1, 1), kernel_regularizer=regularizer, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Conv2D(32, (5, 3), kernel_regularizer=regularizer, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Flatten(input_shape=(WIN, df_train.shape[1], 1)),
#     tf.keras.layers.Dense(128, activation='relu'),
#     # tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Dense(128, activation='relu'),
#     # tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.Dense(2, activation='softmax')
# ])

# loss_fn = tf.keras.losses.mean_squared_error
# loss_fn = tf.keras.losses.CategoricalCrossentropy()
# loss_fn = tf.losses.BinaryCrossentropy()
# loss_fn = WeightedCategoricalCrossEntropy(weights={0: 1, 1: 100 })
loss_fn = tfa.losses.SigmoidFocalCrossEntropy(reduction=tf.keras.losses.Reduction.AUTO)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss=loss_fn,
              metrics=[tf.keras.metrics.Recall(thresholds=0.2, name='recall_0.2'),
                       tf.keras.metrics.Recall(thresholds=0.4, name='recall_0.4'),
                       tf.keras.metrics.Recall(thresholds=0.6, name='recall_0.6'),
                       tf.keras.metrics.Precision(thresholds=0.2, name='precision_0.2'),
                       tf.keras.metrics.Precision(thresholds=0.4, name='precision_0.4'),
                       tf.keras.metrics.Precision(thresholds=0.6, name='precision_0.6')]
              )

# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
#               loss=loss_fn,
#               metrics=[tf.keras.metrics.Recall(thresholds=0.5, class_id=1),
#                        tf.keras.metrics.Precision(thresholds=0.5, class_id=1)])

print(model.summary())

model.fit(train_data_set, steps_per_epoch=len(train_data_set), epochs=5000,
          validation_data=test_data_set, validation_steps=len(test_data_set), verbose=1,
          callbacks=[tf.keras.callbacks.TensorBoard(log_dir='logs', profile_batch=0)]
          )

x, y = next(train_data_set)
res = model(x.astype(np.float32)).numpy()
for r, t in zip(res, y):
    print(r, t)
