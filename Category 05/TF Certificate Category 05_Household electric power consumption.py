"""
ABOUT THE DATASET
Original Source: https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption
The original 'Individual House Hold Electric Power Consumption Dataset' has Measurements of electric power consumption in one household with a one-minute sampling rate over a period of almost 4 years.
Different electrical quantities and some sub-metering values are available.
For the purpose of the examination we have provided a subset containing the data for the first 60 days in the dataset.
We have also cleaned the dataset beforehand to remove missing values. The dataset is provided as a csv file in the project.
The dataset has a total of 7 features ordered by time.

INSTRUCTIONS
Complete the code in following functions:
windowed_dataset()
solution_model()

The model input and output shapes must match the following specifications.
Model input_shape must be (BATCH_SIZE, N_PAST = 24, N_FEATURES = 7), since the testing infrastructure expects a window of past N_PAST = 24 observations of the 7 features to predict the next 24 observations of the same features.
Model output_shape must be (BATCH_SIZE, N_FUTURE = 24, N_FEATURES = 7)
DON'T change the values of the following constants N_PAST, N_FUTURE, SHIFT in the windowed_dataset() BATCH_SIZE in solution_model() (See code for additional note on BATCH_SIZE).
Code for normalizing the data is provided - DON't change it. Changing the normalizing code will affect your score.
HINT: Your neural network must have a validation MAE of approximately 0.055 or less on the normalized validation dataset for top marks.
WARNING: Do not use lambda layers in your model, they are not supported on the grading infrastructure.
WARNING: If you are using the GRU layer, it is advised not to use the 'recurrent_dropout' argument (you can alternatively set it to 0), since it has not been implemented in the cuDNN kernel and may result in much longer training times.
"""

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

def solution():
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

download_and_extract_data()
solution()