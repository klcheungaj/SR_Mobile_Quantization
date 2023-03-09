import tensorflow as tf
import numpy as np
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

input_shape = (1, 28, 28, 1)
# inputs = np.arange(3136, dtype=np.int32).reshape(1, 56, 56, 1)
inputs = np.ones(input_shape, dtype=np.int32)
# x = tf.ones(input_shape)
# x = tf.constant(inputs)
x = tf.convert_to_tensor(inputs, dtype=tf.int32)
x_reshape = tf.transpose(x, [0, 3, 1, 2])

conv1 = tf.keras.layers.Conv2D(1, 5, activation='relu', strides=(1,1), padding="same", input_shape=input_shape[1:], kernel_initializer='OnesV2', dtype='int32', use_bias=False)
y = conv1(x)
y = tf.transpose(y, [0, 3, 1, 2])

np.set_printoptions(threshold=np.inf)
print(x_reshape)
print("=================================================")
print(y)

print(x_reshape[0,0,0:4,0:5].numpy().sum())
print(y[0,0,0,1])

## conclusion: for 56 in_size, 2 stride, same padding. adding 1 layer of 0's at left and top, 2 layers of 0's at right and bottom
## conclusion: for 28 in_size, 1 stride, same padding. adding 2 layers of 0's at left, right, top and bottom