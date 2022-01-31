"""
For this task you will need to train a neural network to predict sunspot activity using the Sunspots.csv provided.
Your neural network is expected to have an MAE of at least 20, with top marks going to one with an MAE of around 15.
At the bottom is provided some testing code should you want to check before uploading which measures the MAE for you.
Strongly recommend you test your model with this to be able to see how it performs.

Sequence(시퀀스)
Sunspots.csv를 사용하여 **태양 흑점 활동(sunspot)을 예측하는 인공신경망을 만듭니다.
MAE 오차 기준으로 최소 20이하로 예측할 것을 권장하며, 탑 랭킹에 들려면 MAE 15 근처에 도달해야합니다.
아래 주어진 샘플코드는 당신의 모델을 테스트 하는 용도로 활용할 수 있습니다.
"""

# Step 00. Import Module
import csv
import tensorflow as tf
import numpy as np
import urllib

from tensorflow.keras.layers import Dense, LSTM, Lambda, Conv1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import Huber

def solution():
    # Step 01. Load DataSet
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/Sunspots.csv'
    urllib.request.urlretrieve(url, 'sunspots.csv')

    # Step 02. Preprocessing
    sunspots = []
    time_step = []
    with open('sunspots.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        # 첫 줄은 header이므로 skip 합니다.
        next(reader)
        for row in reader:
            sunspots.append(float(row[2]))
            time_step.append(int(row[0]))

    series = np.array(sunspots)
    time = np.array(time_step)
    split_time = 3000

    time_train = time[:split_time]
    time_valid = time[split_time:]

    x_train = series[:split_time]
    x_valid = series[split_time:]

    # 윈도우 사이즈
    window_size=30
    # 배치 사이즈
    batch_size = 32
    # 셔플 사이즈
    shuffle_size = 1000

    def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
        series = tf.expand_dims(series, axis=-1)
        ds = tf.data.Dataset.from_tensor_slices(series)
        ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
        ds = ds.flat_map(lambda w: w.batch(window_size + 1))
        ds = ds.shuffle(shuffle_buffer)
        ds = ds.map(lambda w: (w[:-1], w[1:]))
        return ds.batch(batch_size).prefetch(1)

    train_set = windowed_dataset(x_train,
                                 window_size=window_size,
                                 batch_size=batch_size,
                                 shuffle_buffer=shuffle_size)

    validation_set = windowed_dataset(x_valid,
                                      window_size=window_size,
                                      batch_size=batch_size,
                                      shuffle_buffer=shuffle_size)

    # Step 03. Model Define
    model = Sequential([
        tf.keras.layers.Conv1D(60, kernel_size=5,
                             padding="causal",
                             activation="relu",
                             input_shape=[None, 1]),
        tf.keras.layers.LSTM(60, return_sequences=True),
        tf.keras.layers.LSTM(60, return_sequences=True),
        tf.keras.layers.Dense(30, activation="relu"),
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(1),
        tf.keras.layers.Lambda(lambda x: x * 400)
    ])

    # Step 04. Model Compile
    optimizer = SGD(lr=1e-5, momentum=0.9)
    loss= Huber()
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=["mae"])

    # Step 05. Create Checkpoint
    checkpoint_path = 'tmp_checkpoint.ckpt'
    checkpoint = ModelCheckpoint(checkpoint_path,
                                 save_weights_only=True,
                                 save_best_only=True,
                                 monitor='val_mae',
                                 verbose=1)

    # Step 06. Fit
    epochs=100
    history = model.fit(train_set,
                        validation_data=(validation_set),
                        epochs=epochs,
                        callbacks=[checkpoint],
                       )

    # Step 07. Load Weight
    model.load_weights(checkpoint_path)

    return model

solution()