# Local Testing Code
# -- Local envs --
# cpu : Intel(R) Core(TM) i7-10700 CPU
# ram : 32GB
# OS : Windows 11 Home
# gpu : RTX3070

# Step 00. Import Module
import urllib
import os
import zipfile
import pandas as pd

import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv1D, LSTM, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint

# Step 01. Load Dataset
def download_and_extract_data():
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/certificate/household_power.zip'
    urllib.request.urlretrieve(url, 'household_power.zip')
    with zipfile.ZipFile('household_power.zip', 'r') as zip_ref:
        zip_ref.extractall()

download_and_extract_data()

# Step 02. Preprocessing
df = pd.read_csv('household_power_consumption.csv', sep=',', infer_datetime_format=True, index_col='datetime', header=0)

def normalize_series(data, min, max):
    data = data - min
    data = data / max
    return data

# FEATURES에 데이터프레임의 Column 개수 대입
N_FEATURES = len(df.columns)

# 데이터프레임을 numpy array으로 가져와 data에 대입
data = df.values

# 데이터 정규화
data = normalize_series(data, data.min(axis=0), data.max(axis=0))

# 데이터셋 분할 (0.8).
# 기존 0.5 -> 0.8로 변경 // 다른 비율로 변경 가능
split_time = int(len(data) * 0.8)
x_train = data[:split_time]
x_valid = data[split_time:]

def windowed_dataset(series, batch_size, n_past=24, n_future=24, shift=1):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(size=(n_past + n_future), shift = shift, drop_remainder = True)
    ds = ds.flat_map(lambda w: w.batch(n_past + n_future))
    ds = ds.shuffle(len(series))
    ds = ds.map(
        lambda w: (w[:n_past], w[n_past:])
    )
    return ds.batch(batch_size).prefetch(1)

# 다음 4개의 옵션은 주어 집니다.
BATCH_SIZE = 32 # 변경 가능하나 더 올리는 것은 비추 (내리는 것은 가능하나 시간 오래 걸림)
N_PAST = 24 # 변경 불가.
N_FUTURE = 24 # 변경 불가.
SHIFT = 1 # 변경 불가.

train_set = windowed_dataset(series=x_train,
                             batch_size=BATCH_SIZE,
                             n_past=N_PAST,
                             n_future=N_FUTURE,
                             shift=SHIFT)

valid_set = windowed_dataset(series=x_valid,
                             batch_size=BATCH_SIZE,
                             n_past=N_PAST,
                             n_future=N_FUTURE,
                             shift=SHIFT)

# Step 03. Model Define
model = tf.keras.models.Sequential([
    Conv1D(filters=32,
            kernel_size=3,
            padding="causal",
            activation="relu",
            input_shape=[N_PAST, 7],
            ),
    Bidirectional(LSTM(32, return_sequences=True)),
    Dense(32, activation="relu"),
    Dense(16, activation="relu"),
    Dense(N_FEATURES)
])

# Step 04. Create Checkpoint
checkpoint_path='household/my_checkpoint.ckpt'
checkpoint = ModelCheckpoint(checkpoint_path,
                             save_weights_only=True,
                             save_best_only=True,
                             monitor='val_loss',
                             verbose=1,
                             )

# Step 05. Model Compile
# learning_rate=0.0005, Adam 옵치마이저
optimizer =  tf.keras.optimizers.Adam(learning_rate=0.0005)

model.compile(loss='mae',
              optimizer=optimizer,
              metrics=["mae"]
              )

# Step 06. Fit
model.fit(train_set,
        validation_data=(valid_set),
        epochs=20,
        callbacks=[checkpoint],
        )

# Step 07. Load Weight
model.load_weights(checkpoint_path)

# Extra Step. Validation Model
model.evaluate(valid_set)