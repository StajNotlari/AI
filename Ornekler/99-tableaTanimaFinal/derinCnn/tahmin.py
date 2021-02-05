import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.get_logger().setLevel('FATAL')

from skimage import io
from skimage import transform
from skimage import exposure

import numpy as np

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer
from tensorflow.python.keras.layers import MaxPooling2D
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten
from tensorflow.python.keras.optimizers import Adam

import gtsrb


def load_image(image_path):
    image = io.imread(image_path)
    image = transform.resize(image, (32, 32))
    image = exposure.equalize_adapthist(image, clip_limit=0.1)
    image = np.array(image)

    return image


class_names = gtsrb.get_class_names()

image = load_image("D:/Isler/Yapay Zeka/GitHub/Ornekler/99-tableaTanimaFinal/derinCnn/test/2.jpg")
image = image.reshape([1, 32, 32, 3])

numLabels = 43

model = Sequential()

model.add(InputLayer(input_shape=(32, 32, 3)))

model.add(Conv2D(kernel_size=5, strides=1, filters=16, padding='same', activation='relu', name='conv1'))
model.add(MaxPooling2D(pool_size=2, strides=2))

model.add(Conv2D(kernel_size=5, strides=1, filters=32, padding='same', activation='relu', name='conv2'))
model.add(MaxPooling2D(pool_size=2, strides=2))

model.add(Flatten())

model.add(Dense(8 * 8 * 32, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(numLabels, activation='softmax'))

optimizer = Adam(lr=1e-3)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint_path = "D:/Isler/Yapay Zeka/GitHub/Ornekler/99-tableaTanimaFinal/derinCnn/checkpoints/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)


latest = tf.train.latest_checkpoint(checkpoint_dir)
if latest != None:
    print("model agirliklari yuklendi")
    model.load_weights(latest)
else:
    print("model agirliklari yok. tahmin yapilamaz")
    os._exit()


result = model.predict_classes(image)
print("Tahmin: " + class_names[result[0]])