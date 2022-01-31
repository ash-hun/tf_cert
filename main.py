# Local Testing Code
# -- Local envs --
# cpu : Intel(R) Core(TM) i7-10700 CPU
# ram : 32GB
# OS : Windows 11 Home
# gpu : RTX3070

import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
print(tf.__version__)
print(sys.version)

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=float)

model = Sequential([
    Dense(1, input_shape=[1]),
])

model.compile(optimizer='sgd', loss='mse')
model.fit(xs, ys, epochs=100, verbose=0)

print(f"검증완료 : {model.predict([10.0])}")