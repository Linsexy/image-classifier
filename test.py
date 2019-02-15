from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


train_images = train_images / 255.0

test_images = test_images / 255.0


train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(-1, 28, 28, 1)


# model = keras.Sequential([
#    keras.layers.Flatten(input_shape=(28, 28)),
#    keras.layers.Dense(128, activation=tf.nn.relu),
#    keras.layers.Dense(10, activation=tf.nn.softmax)
#])

model = keras.Sequential([
	keras.layers.Conv2D(6, (4, 4), input_shape=(28, 28, 1)),
	keras.layers.PReLU(),
	keras.layers.MaxPooling2D(2),
	keras.layers.Flatten(),
	keras.layers.Dense(294),
	keras.layers.Dense(10, activation=tf.nn.softmax)
	])

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)



