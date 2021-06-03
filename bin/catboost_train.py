from datetime import date
from tqdm import tqdm

import numpy as np
import pandas as pd

from binance_prediction.data_generator import DataGenerator
from binance_prediction.feature_engineering import feature_engineering
from binance_prediction.pre_processing import load_pre_processing, resample

from catboost import CatBoostClassifier

path = '/home/serg/DataspellProjects/binance/BTCUSDT_2017-08-17_2021-05-21.csv'

df = pd.read_csv(path)
df = load_pre_processing(df)


df = resample(df, '60Min')

# FILTER BY 2018+
df = df[df.index > pd.Timestamp(date(2018, 1, 1))]

df = feature_engineering(df)

# TRAIN/TEST SPLIT
df_train = df[df.index < pd.Timestamp(date(2021, 1, 1))]
df_test = df[df.index >= pd.Timestamp(date(2021, 1, 1))]


WIN = 30
train_batch_size = 1
test_batch_size = 1

train_data_set = DataGenerator(df_train, batch_size=train_batch_size, win=WIN, phase='train')
test_data_set = DataGenerator(df_test, batch_size=test_batch_size, win=WIN, phase='test')

print(f'train length: {len(df_train)}', f'batch_size: {train_batch_size}', f'samples:{len(df_train)/train_batch_size}')
print(f'test length: {len(df_test)}', f'batch_size: {test_batch_size}', f'samples:{len(df_test)/test_batch_size}')

print('TRAIN: count of pump candles', df_train['two_percent_pump'].sum(), 'common count', len(df_train))
print('TEST: count of pump candles', df_test['two_percent_pump'].sum(), 'common count', len(df_test))


def get_set(data_set, batch_size):
    x_set = list()
    y_set = list()
    for i in tqdm(range(int(len(data_set.indexes)/batch_size))):
        x, y = next(data_set)
        y = np.argmax(y, axis=-1)
        x_set.append(x)
        y_set.append(y)
    x_set = np.concatenate(x_set, axis=0)
    y_set = np.concatenate(y_set, axis=0)
    return np.reshape(x_set, newshape=(x_set.shape[0], x_set.shape[1]*x_set.shape[2]*x_set.shape[3])), y_set


x_train, y_train = get_set(train_data_set, train_batch_size)
x_test, y_test = get_set(test_data_set, test_batch_size)

print(x_train.shape)

count = np.bincount(y_train.astype('int32'))
median = np.median(count)
min_ = np.min(count)
weights = median / count
print(weights)

model = CatBoostClassifier(iterations=200,
                           learning_rate=0.1,
                           depth=6,
                           loss_function='MultiClassOneVsAll',
                           eval_metric='MultiClassOneVsAll',
                           # class_weights=weights,
                           class_weights=[0.4, 1],
                           task_type="GPU",
                           devices='0'
                           )

model.fit(x_train, y_train, early_stopping_rounds=20, eval_set=(x_test, y_test), use_best_model=True)

# cm = confusion_matrix(y_test, model.predict(x_test), labels=list(crop_legend_dict.keys()))
# cm_image = plot_cm(cm, 'cm', len(crop_legend_dict), legend=crop_legend_dict)
# fig = plt.figure(figsize=(50, 50))
# plt.imshow(cm_image)
# plt.show()

importance = model.get_feature_importance()
pred = model.predict(x_test)
print(len(y_test))
print()
for p, y in zip(pred, y_test):
    if y == 1 or p[0] == 1:
        print(y, p[0])

from sklearn.metrics import precision_score, recall_score


print(precision_score(y_test, pred[..., 0], average=None))
print(recall_score(y_test, pred[..., 0], average=None))
print('pred sum: ', sum(pred[..., 0]))
print(len(y_test))

