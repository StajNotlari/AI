import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data =keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=10000)

word_index = data.get_word_index()

word_index = { k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"],padding="post", maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"],padding="post", maxlen=250)

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_rewiew(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])

print(decode_rewiew(test_data[2]))
print(test_labels[2])


model = keras.models.load_model("model2.h5")

result = model.evaluate(test_data, test_labels)
print(result)

model.save("model2.h5")