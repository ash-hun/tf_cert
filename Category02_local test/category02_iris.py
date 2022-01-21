# Local Testing Code
# -- Local envs --
# cpu : Intel(R) Core(TM) i7-10700 CPU
# ram : 32GB
# OS : Windows 11 Home
# gpu : RTX3070

# 필요한 패키지 import
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint

# STEP 2: 데이터 전처리
# data = tfds.load("iris", split=tfds.Split.TRAIN.subsplit(tfds.percent[:80]))
train_dataset = tfds.load('iris', split='train[:80%]')
valid_dataset = tfds.load('iris', split='train[80%:]')

# 전처리 요구조건
# One-hot encoding 할것.
# feature(x)와 label(y)를 분리할 것

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

# STEP 3: 모델의 정의 (modeling)
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

# STEP 4: 모델의 생성 (compile)
model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics=['acc'])


checkpoint_path="my_checkpoint_iris.ckpt"
checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                             save_weights_only=True,
                             save_best_only=True,
                             monitor='val_loss',
                             verbose=1)
# STEP 5: 학습 (fit)
history = model.fit(train_data,
                    validation_data=(valid_data),
                    epochs=20,
                    callbacks=[checkpoint],
                    )

model.load_weights(checkpoint_path)