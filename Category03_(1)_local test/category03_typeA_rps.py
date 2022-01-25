# Local Testing Code
# -- Local envs --
# cpu : Intel(R) Core(TM) i7-10700 CPU
# ram : 32GB
# OS : Windows 11 Home
# gpu : RTX3070

# Step 00. Import Module
import urllib.request
import zipfile
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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

# Step 01. Load Dataset
url = 'https://storage.googleapis.com/download.tensorflow.org/data/rps.zip'
urllib.request.urlretrieve(url, 'rps.zip')
local_zip = 'rps.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('tmp/')
zip_ref.close()

# Step 02. Data Preprocessing
TRAINING_DIR = "./tmp/rps/"

training_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
    )

training_generator = training_datagen.flow_from_directory(TRAINING_DIR,
                                                          batch_size=128,
                                                          target_size=(150, 150),
                                                          class_mode='categorical',
                                                          subset='training',
                                                         )
validation_generator = training_datagen.flow_from_directory(TRAINING_DIR,
                                                          batch_size=128,
                                                          target_size=(150, 150),
                                                          class_mode='categorical',
                                                          subset='validation',
                                                         )

# Step 03. Model Define
model = Sequential([
    # Conv2D, MaxPooling2D 조합으로 층을 쌓습니다. 첫번째 입력층의 input_shape은 (150, 150, 3)으로 지정합니다.
    Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dense(3, activation='softmax'),
])

# Step 04. Model Compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

# Step 05. Model Checkpoint
checkpoint_path = "tmp_checkpoint.ckpt"
checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                             save_weights_only=True,
                             save_best_only=True,
                             monitor='val_loss',
                             verbose=1)

# Step 06. Fit
model.fit(training_generator,
          validation_data=(validation_generator),
          epochs=25,
          callbacks=[checkpoint],
          )

# Step 07. Load Weight
model.load_weights(checkpoint_path)