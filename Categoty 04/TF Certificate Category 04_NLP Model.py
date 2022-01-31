"""
NLP QUESTION

For this task you will build a classifier for the sarcasm dataset The classifier should have a final layer with 1 neuron activated by sigmoid as shown.

It will be tested against a number of sentences that the network hasn't previously seen
And you will be scored on whether sarcasm was correctly detected in those sentences

자연어 처리

이 작업에서는 sarcasm 데이터 세트에 대한 분류기를 작성합니다. 분류기는 1 개의 뉴런으로 이루어진 sigmoid 활성함수로 구성된 최종 층을 가져야합니다.
제출될 모델은 데이터셋이 없는 여러 문장에 대해 테스트됩니다. 그리고 당신은 그 문장에서 sarcasm 판별이 제대로 감지되었는지에 따라 점수를 받게 될 것입니다
"""

# Step 00. Import Module
import json
import tensorflow as tf
import numpy as np
import urllib

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:  # gpu가 있다면, 용량 한도를 5GB로 설정
#   print("gpu set")
#   tf.config.experimental.set_virtual_device_configuration(gpus[0],
#         [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5*1024)])
if gpus:
  try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
  except RuntimeError as e:
    # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
    print(e)

# Step 01. Load DataSet
url = 'https://storage.googleapis.com/download.tensorflow.org/data/sarcasm.json'
urllib.request.urlretrieve(url, 'sarcasm.json')

with open('sarcasm.json') as f:
    datas = json.load(f)

# Step 02. Preprocessing Data
sentences = []
labels = []
for data in datas:
    sentences.append(data['headline'])
    labels.append(data['is_sarcastic'])

training_size = 20000
train_sentences = sentences[:training_size]
train_labels = labels[:training_size]
validation_sentences = sentences[training_size:]
validation_labels = labels[training_size:]

vocab_size = 1000
oov_tok = "<OOV>"
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)

tokenizer.fit_on_texts(train_sentences)

train_sequences = tokenizer.texts_to_sequences(train_sentences)
validation_sequences = tokenizer.texts_to_sequences(validation_sentences)

# 한 문장의 최대 단어 숫자
max_length = 120

# 잘라낼 문장의 위치
trunc_type='post'

# 채워줄 문장의 위치
padding_type='post'

train_padded = pad_sequences(train_sequences, maxlen=max_length, truncating=trunc_type, padding=padding_type)
validation_padded = pad_sequences(validation_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

train_labels = np.array(train_labels)
validation_labels = np.array(validation_labels)

# Step 02-2. Embedding Layer
embedding_dim = 16

# Step 03. Model Define
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_length),
    Bidirectional(LSTM(64, return_sequences=True)),
    Bidirectional(LSTM(64, return_sequences=True)),
    Bidirectional(LSTM(64, return_sequences=True)),
    Bidirectional(LSTM(64)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Step 04. Model Compile
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

# Step 05. Create Checkpoint
checkpoint_path = 'my_checkpoint.ckpt'
checkpoint = ModelCheckpoint(checkpoint_path,
                             save_weights_only=True,
                             save_best_only=True,
                             monitor='val_loss',
                             verbose=1)

# Step 05. Fit
epochs=10
history = model.fit(train_padded, train_labels,
                    validation_data=(validation_padded, validation_labels),
                    callbacks=[checkpoint],
                    epochs=epochs)


# Step 06. Load Weight
model.load_weights(checkpoint_path)
