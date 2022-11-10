# Get the MNIST dataset
import numpy as np 
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

print(x_train.shape, x_test.shape)

# make the labels be 1 if the digit is a 0, and 0 otherwise
y_train = (y_train == 0).astype(np.float32)
y_test = (y_test == 0).astype(np.float32)

# normalize each training row to have unit norm
x_train = x_train / np.mean(np.linalg.norm(x_train, axis=1))
x_test = x_test / np.mean(np.linalg.norm(x_test, axis=1))