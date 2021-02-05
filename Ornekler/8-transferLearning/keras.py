import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import numpy as np
import time
import matplotlib.pyplot as plt
from datetime import timedelta

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer
from tensorflow.python.keras.layers import  MaxPooling2D
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten
from tensorflow.python.keras.optimizers import Adam

import cifar10
from cifar10 import num_classes
import inception
from inception import transfer_values_cache

cifar10.download()
inception.download()

model = inception.Inception()

train_img, train_cls, train_labels = cifar10.load_training_data()#resimler | 50000*32*32*3, siniflari | 50000*1, softmax_halinde siniflari [0, 0, 1, 0, ...] | 50000*10
test_img, test_cls, test_labels = cifar10.load_test_data()

file_path_cache_train = os.path.join(cifar10.data_path, 'inception_cifar10_train.pkl')
file_path_cache_test = os.path.join(cifar10.data_path, 'inception_cifar10_test.pkl')

images_scaled = train_img * 255.0 #cifar piksel renk datası 0-1 arası. bunu 0-255 arası hale getiriyoruz

#inception modelinin son katmanını siler. gönderdiğin resimleri modelden geçirir. burda dönen data eğitilmiş modelden geçmiş data
#sonra biz bu datayı eğitilmemiş iki katmana sokuyoruz ve onu eğitiyoruz.
transfer_values_train = transfer_values_cache(cache_path=file_path_cache_train,
                                              images=images_scaled,
                                              model=model)


images_scaled = test_img * 255.0
transfer_values_test = transfer_values_cache(cache_path=file_path_cache_test,
                                              images=images_scaled,
                                              model=model)

print(transfer_values_train.shape, num_classes)#50000*2048  10 #50000 resim var. Inception; son katman çıkarılınca da 2048 tane çıkış veriyor

model = Sequential()

model.add(InputLayer(input_shape=(2048)))
model.add(Dense(1024, activation='relu'))
model.add(Dense(10, activation='softmax'))

optimizer = Adam(lr=1e-3)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x=transfer_values_train, y=train_labels, epochs=10, batch_size=128)
result = model.evaluate(x=transfer_values_test, y=test_labels)
print('\n\nAccuracy:', result[1])