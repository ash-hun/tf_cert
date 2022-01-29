"""
For this task you will build a classifier for Rock-Paper-Scissors based on the rps dataset.
IMPORTANT: Your final layer should be as shown, do not change the provided code, or the tests may fail
IMPORTANT: Images will be tested as 150x150 with 3 bytes of color depth So ensure that your input layer is designed accordingly, or the tests may fail.
NOTE THAT THIS IS UNLABELLED DATA. You can use the ImageDataGenerator to automatically label it and we have provided some starter code.
이 작업에서는 Rock-Paper-Scissors에 대한 분류기를 작성합니다. rps 데이터 셋을 기반으로합니다.
중요 : 최종 레이어는 그림과 같아야합니다.
중요 : 이미지는 3 바이트 150x150의 컬러사진으로 테스트됩니다. 따라서 입력 레이어가 그에 따라 설계되었거나 테스트되었는지 확인하십시오.
ImageDataGenerator를 사용하여 자동으로 레이블을 지정할 수 있습니다.
"""

# Step 00. Import Module
import urllib.request
import zipfile
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint


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