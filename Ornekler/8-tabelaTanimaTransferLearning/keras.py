import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import numpy as np

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer
from tensorflow.python.keras.layers import MaxPooling2D
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten
from tensorflow.python.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

import inception
from inception import transfer_values_cache

import gtsrb

file_path_cache_train = os.path.join(gtsrb.data_path, 'inception_gtsrb_train.pkl')
file_path_cache_test = os.path.join(gtsrb.data_path, 'inception_gtsrb_test.pkl')

[trainX, trainY, testX, testY] = gtsrb.get_data()

numLabels = len(np.unique(trainY))

trainY = to_categorical(trainY, numLabels)
testY = to_categorical(testY, numLabels)

inception.download()
model = inception.Inception()


transfer_values_train = transfer_values_cache(cache_path=file_path_cache_train,
                                              images=trainX,
                                              model=model)


transfer_values_test = transfer_values_cache(cache_path=file_path_cache_test,
                                              images=testX,
                                              model=model)

print(transfer_values_train.shape, numLabels)#39209*2048  43 #39209 resim var. Inception; son katman çıkarılınca da 2048 tane çıkış veriyor

model = Sequential()

model.add(InputLayer(input_shape=(2048)))

"""model.add(Conv2D(kernel_size=5, strides=1, filters=16, padding='same', activation='relu', name='conv1'))
model.add(MaxPooling2D(pool_size=2, strides=2))#boyut tam yariya duser

model.add(Conv2D(kernel_size=5, strides=1, filters=32, padding='same', activation='relu', name='conv2'))
model.add(MaxPooling2D(pool_size=2, strides=2))

model.add(Flatten())"""

#model.add(Dense(8 * 8 * 32, activation='relu'))
#model.add(Dense(512, activation='relu'))
model.add(Dense(numLabels, activation='softmax'))



optimizer = Adam(lr=1e-5)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


checkpoint_path = "D:/Isler/Yapay Zeka/GitHub/Ornekler/8-tabelaTanimaTransferLearning/checkpoints/cp-{epoch:04d}.ckpt"
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


model.fit(x=transfer_values_train, y=trainY, epochs=20, batch_size=128, callbacks = [cp_callback])


result = model.evaluate(x=transfer_values_test, y=testY)
print('\n\nAccuracy:', result[1])