# Local Testing Code
# -- Local envs --
# cpu : Intel(R) Core(TM) i7-10700 CPU
# ram : 32GB
# OS : Windows 11 Home
# gpu : RTX3070

"""
Create a classifier for the Fashion MNIST dataset
Note that the test will expect it to classify 10 classes and that
the input shape should be the native size of the Fashion MNIST dataset which is 28x28 monochrome.
Do not resize the data. Your input layer should accept
(28,28) as the input shape only.
If you amend this, the tests will fail.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint

fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_valid, y_valid) = fashion_mnist.load_data()

x_train = x_train / 255.0
x_valid = x_valid / 255.0

# 시각화
fig, axes = plt.subplots(2, 5)
fig.set_size_inches(10, 5)

for i in range(10):
    axes[i//5, i%5].imshow(x_train[i], cmap='gray')
    axes[i//5, i%5].set_title(str(y_train[i]), fontsize=15)
    plt.setp( axes[i//5, i%5].get_xticklabels(), visible=False)
    plt.setp( axes[i//5, i%5].get_yticklabels(), visible=False)
    axes[i//5, i%5].axis('off')

plt.tight_layout()
plt.show()

tf.keras.backend.set_floatx('float64')
x = Flatten(input_shape=(28, 28))
print(x(x_train).shape)
