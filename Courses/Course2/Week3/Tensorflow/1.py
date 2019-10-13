import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data =keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()


#plt.imshow(tarin_images[1], cmap=plt.cm.binary)
#plt.show()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0
test_images = test_images / 255.0


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(train_images, train_labels, epochs=1)



#test_loss, test_acc = model.evaluate(test_images, test_labels)
#print(test_acc)

#toplu tahmin
predict = model.predict(test_images)
print(predict.shape)
print(np.argmax(predict[0]))
print(class_names[np.argmax(predict[0])])


for i in range(2):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Doğru:" + class_names[np.argmax(test_labels[i])])
    plt.title("Tahmin: " + class_names[np.argmax(predict[i])] )
    plt.show()

