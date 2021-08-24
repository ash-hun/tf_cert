import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=float)

# model = Sequential([
#     Dense(1, input_shape=[1]),
#     Dense(4),
#     Dense(4),
#     Dense(1)
# ])

model = Sequential([
    Dense(1, input_shape=[1]),
])

model.compile(optimizer='sgd', loss='mse')

model.fit(xs, ys, epochs=1200, verbose=0)

print(model.predict([10.0]))
