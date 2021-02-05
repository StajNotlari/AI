import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.get_logger().setLevel('FATAL')

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten
from tensorflow.python.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

import numpy as np

import gtsrb

[trainX, trainY, testX, testY] = gtsrb.get_data()

numLabels = len(np.unique(trainY))

trainY = to_categorical(trainY, numLabels)
testY = to_categorical(testY, numLabels)


model = Sequential()

model.add(InputLayer(input_shape=(32, 32, 3)))
model.add(Flatten())
model.add(Dense(32*32*3, activation='relu'))
model.add(Dense(numLabels, activation='softmax'))

optimizer = Adam(lr=1e-3)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x=trainX, y=trainY, epochs=10, batch_size=128)
result = model.evaluate(x=testX, y=testY)
print('\n\nAccuracy:', result[1])