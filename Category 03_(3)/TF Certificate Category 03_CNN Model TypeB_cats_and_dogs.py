"""
Computer Vision with CNNs
For this exercise you will build a cats v dogs classifier
using the Cats v Dogs dataset from TFDS.
Be sure to use the final layer as shown
(Dense, 2 neurons, softmax activation)
The testing infrastructre will resize all images to 224x224
with 3 bytes of color depth. Make sure your input layer trains
images to that specification, or the tests will fail.
Make sure your output layer is exactly as specified here, or the
tests will fail.

이 연습에서는 cats v dogs 분류기를 만들 것입니다. TFDS의 Cats v Dogs 데이터 세트 사용.
그림과 같이 최종 레이어를 사용하십시오
(Dense, 뉴런 2 개, activation='softmax')
테스트 인프라는 모든 이미지의 크기를 224x224로 조정합니다(컬러사진). 입력 레이어를 확인하십시오
"""

# Step 00. Import Module
import tensorflow_datasets as tfds
import tensorflow as tf

from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import VGG16
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
dataset_name = 'cats_vs_dogs'

# 처음 80%의 데이터만 사용
train_dataset = tfds.load(name=dataset_name, split='train[:80%]')

# 최근 20%의 데이터만 사용
valid_dataset = tfds.load(name=dataset_name, split='train[80%:]')

# Step 02. Data Preprocessing
def preprocess(data):
  x = data['image']/255
  y = data['label']
  x = tf.image.resize(x, size=(224, 224))
  return x, y

batch_size=32
train_data = train_dataset.map(preprocess).batch(batch_size)
valid_data = valid_dataset.map(preprocess).batch(batch_size)

# Step 03. Model Define
# Transfer Learning(전이학습)
transfer_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
transfer_model.trainable=False

model = Sequential([
    transfer_model,
    Flatten(),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dense(128, activation='relu'),
    Dense(2, activation='softmax'),
])

# Step 04. Model Compile
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])

# Step 05. Create Model Checkpoint
checkpoint_path = "my_checkpoint.ckpt"
checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                             save_weights_only=True,
                             save_best_only=True,
                             monitor='val_loss',
                             verbose=1)

# Step 06. Fit
model.fit(train_data,
          validation_data=(valid_data),
          epochs=20,
          callbacks=[checkpoint],
          )

# Step 07. Load Weights
model.load_weights(checkpoint_path)