import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Lambda, Dropout, BatchNormalization
from tensorflow.python.keras.layers import MaxPooling2D
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

model.add(Conv2D(kernel_size=3, strides=1, filters=32, padding='same', activation='relu', name='conv1'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=2, strides=2))

model.add(Conv2D(kernel_size=3, strides=1, filters=64, padding='same', activation='relu', name='conv2'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=2, strides=2))

model.add(Conv2D(kernel_size=3, strides=1, filters=64, padding='same', activation='relu', name='conv3'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=2, strides=2))

model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.75))

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.75))

model.add(Dense(numLabels, activation='softmax'))


optimizer = Adam(lr=1e-3)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


checkpoint_path = "D:/Isler/Yapay Zeka/GitHub/Ornekler/6-tabelaTanimaDataAugDropoutBatchNormModelKaydet/checkpoints/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)


latest = tf.train.latest_checkpoint(checkpoint_dir)
if latest != None:
    print("model agirliklari yuklendi")
    print(latest)
    model.load_weights(latest)
else:
    print("model agirliklari yok. egitim sifirdan baslayacak")


cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, verbose=1, save_weights_only=True, period=1)
model.save_weights(checkpoint_path.format(epoch=0))


model.fit(x=trainX, y=trainY, epochs=1, batch_size=128, callbacks = [cp_callback])

result = model.evaluate(x=testX, y=testY)
print('\n\nAccuracy:', result[1])