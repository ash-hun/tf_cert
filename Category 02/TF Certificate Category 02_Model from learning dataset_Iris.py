"""
For this task you will train a classifier for Iris flowers using the Iris dataset
The final layer in your neural network should look like: tf.keras.layers.
Dense(3, activation=tf.nn.softmax)
The input layer will expect data in the shape (4,)
We've given you some starter code for preprocessing the data
You'll need to implement the preprocess function for data.map

이 작업에서는 Iris 데이터 세트를 사용하여 Iris 꽃 분류기를 훈련시킵니다.
신경망의 마지막 계층은 tf.keras.layers와 같아야합니다.
Dense(3, activation='softmax')
입력 레이어는 모양의 데이터를 기대합니다 (4,)
데이터 전처리를위한 스타터 코드를 제공했습니다
data.map에 대한 전처리 기능을 구현해야합니다.
"""

# Third Party Module import
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint

# Data preprocessing
# data = tfds.load("iris", split=tfds.Split.TRAIN.subsplit(tfds.percent[:80]))
train_dataset = tfds.load('iris', split='train[:80%]')
valid_dataset = tfds.load('iris', split='train[80%:]')


def preprocess(data):
    # write your code
    x = data['features']
    y = data['label']
    y = tf.one_hot(y, 3)
    return x, y

for data in train_dataset.take(5):
    preprocess(data)

batch_size = 10 #주어지는 값!
train_data = train_dataset.map(preprocess).batch(batch_size)
valid_data = valid_dataset.map(preprocess).batch(batch_size)

# Modeling
model = tf.keras.models.Sequential([
    # input_shape는 X의 feature 갯수가 4개 이므로 (4, )로 지정합니다.
    Dense(512, activation='relu', input_shape=(4, )),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    # Classification을 위한 Softmax, 클래스 갯수 = 3개
    Dense(3, activation='softmax'),
])

# Model Compile
model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics=['acc'])

# Create Model CheckPoint
checkpoint_path="my_checkpoint_iris.ckpt"
checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                             save_weights_only=True,
                             save_best_only=True,
                             monitor='val_loss',
                             verbose=1)

# Fit
history = model.fit(train_data,
                    validation_data=(valid_data),
                    epochs=20,
                    callbacks=[checkpoint],
                    )

# Load Weights after fit
model.load_weights(checkpoint_path)