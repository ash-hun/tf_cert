"""
Build and train a neural network to predict
the time indexed variable of the univariate US diesel prices (On - Highway) All types for the period of 1994 - 2021.
Using a window of past 10 observations of 1 feature , train the model to predict the next 10 observations of that feature.

HINT:
If you follow all the rules mentioned above and throughout this question while training your neural network,
there is a possibility that a validation MAE of approximately 0.02 or less on the normalized validation dataset may fetch you top marks.
"""

# Step 00. Import Module
import urllib
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv1D, LSTM, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint

# Step 01. Load Dataset
# 시험때는 아래 2줄 코드는 넣지 말아 주세요!!!
url = 'https://www.dropbox.com/s/eduk281didil1km/Weekly_U.S.Diesel_Retail_Prices.csv?dl=1'
urllib.request.urlretrieve(url, 'Weekly_U.S.Diesel_Retail_Prices.csv')

# Step 02. Preprocessing
# This function normalizes the dataset using min max scaling.
# DO NOT CHANGE THIS CODE
def normalize_series(data, min, max):
    data = data - min
    data = data / max
    return data

# DO NOT CHANGE THIS.
def windowed_dataset(series, batch_size, n_past=10, n_future=10, shift=1):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(size=n_past + n_future, shift=shift, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(n_past + n_future))
    ds = ds.map(lambda w: (w[:n_past], w[n_past:]))
    return ds.batch(batch_size).prefetch(1)

def solution():
    df = pd.read_csv('Weekly_U.S.Diesel_Retail_Prices.csv', infer_datetime_format=True, index_col='Week of', header=0)
    N_FEATURES = len(df.columns)

    # 정규화 코드
    data = df.values
    data = normalize_series(data, data.min(axis=0), data.max(axis=0))

    # 데이터 분할
    SPLIT_TIME = int(len(data) * 0.8) # DO NOT CHANGE THIS
    x_train = data[:SPLIT_TIME]
    x_valid = data[SPLIT_TIME:]

    BATCH_SIZE = 32  # 배치사이즈
    N_PAST = 10      # 과거 데이터 (X)
    N_FUTURE = 10    # 미래 데이터 (Y)
    SHIFT = 1        # SHIFT

    train_set = windowed_dataset(series=x_train, batch_size=BATCH_SIZE,
                                 n_past=N_PAST, n_future=N_FUTURE,
                                 shift=SHIFT)

    valid_set = windowed_dataset(series=x_valid, batch_size=BATCH_SIZE,
                                 n_past=N_PAST, n_future=N_FUTURE,
                                 shift=SHIFT)

    # Step 03. Model Define
    model = tf.keras.models.Sequential([
        Conv1D(filters=32, kernel_size=5, padding='causal', activation='relu', input_shape=[N_PAST, 1]),
        Bidirectional(LSTM(32, return_sequences=True)),
        Bidirectional(LSTM(32, return_sequences=True)),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(N_FEATURES)
    ])

    # Step 04. Create Checkpoint
    checkpoint_path = 'USretail/my_checkpoint.ckpt'
    checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                                 save_weights_only=True,
                                 save_best_only=True,
                                 monitor='val_mae',
                                 verbose=1)

    # Step 05. Model Compile
    optimizer = tf.keras.optimizers.Adam(0.0001)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.Huber(), metrics=['mae'])

    # Step 06. Fit
    model.fit(train_set,
              validation_data=(valid_set),
              epochs=100,
              callbacks=[checkpoint])

    # Step 07. Load Weight
    model.load_weights(checkpoint_path)

solution()