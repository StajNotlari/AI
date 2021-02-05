import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Dropout, BatchNormalization
from tensorflow.python.keras.layers import MaxPooling2D
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten
from tensorflow.python.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from tensorflow.python.keras.datasets import cifar10

import numpy as np

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('Training set:', len(x_train), 'Testing set:', len(x_test))

numLabels = len(np.unique(y_train))

y_train = to_categorical(y_train, numLabels)
y_test = to_categorical(y_test, numLabels)

#img_rows, img_cols , channels= 32,32,3
"""for i in range(0,9):
    plt.subplot(330 + 1 + i)
    plt.imshow(x_train[i])
plt.show()"""


datagen = ImageDataGenerator(
    rotation_range=15,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
    #zoom_range=0.3
    )
datagen.fit(x_train)

"""for X_batch, y_batch in datagen.flow(x_train, y_train, batch_size=9):
    for i in range(0, 9):
        plt.subplot(330 + 1 + i)
        plt.imshow(X_batch[i].astype(np.uint8))
    plt.show()
    break"""


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
model.add(Dropout(0.5))

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))


optimizer = Adam(lr=1e-3)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])



checkpoint_path = "D:/Isler/Yapay Zeka/GitHub/Ornekler/6-modelKaydet/keras_checkpoints/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True,
    period=1)

model.save_weights(checkpoint_path.format(epoch=0))



model.fit(x=x_train, y=y_train, epochs=1, batch_size=128, callbacks = [cp_callback])


result = model.evaluate(x=x_test, y=y_test)
print('\n\nAccuracy:', result[1])