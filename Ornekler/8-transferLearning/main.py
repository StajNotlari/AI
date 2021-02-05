import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import numpy as np
import time
import matplotlib.pyplot as plt
from datetime import timedelta

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

x = tf.placeholder(tf.float32, [None, 2048])
y_true = tf.placeholder(tf.float32, [None, num_classes])

weight1 = tf.Variable(tf.truncated_normal([2048, 1024], stddev=0.1))
bias1 = tf.Variable(tf.constant(0.1, shape=[1024]))
weight2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
bias2 = tf.Variable(tf.constant(0.1, shape=[10]))

y1 = tf.nn.relu(tf.matmul(x, weight1) + bias1)
logits = tf.matmul(y1, weight2) + bias2
y2 = tf.nn.softmax(logits)
y_pred_cls = tf.argmax(y2, 1)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true)
loss = tf.reduce_mean(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y2, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

optimizer = tf.train.AdamOptimizer(5e-4).minimize(loss)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

batch_size = 128
def random_batch():
    num_images = len(transfer_values_train)
    idx = np.random.choice(num_images, size=batch_size, replace=False)
    x_batch = transfer_values_train[idx]
    y_batch = train_labels[idx]

    return x_batch, y_batch


loss_graph = []
def training_step (iterations):
    start_time = time.time()
    for i in range (iterations):
        x_batch, y_batch = random_batch()
        feed_dict_train = {x: x_batch, y_true: y_batch}
        [_, train_loss] = sess.run([optimizer, loss], feed_dict=feed_dict_train)
        loss_graph.append(train_loss)

        if i % 100 == 0:
            acc = sess.run(accuracy, feed_dict=feed_dict_train)
            print('Iteration:', i, 'Training accuracy:', acc, 'Training loss:', train_loss)

    end_time = time.time()
    time_dif = end_time - start_time
    print("Time usage: ", timedelta(seconds=int(round(time_dif))))


batch_size_test = 256
def test_accuracy():
    num_images = len(transfer_values_test)
    cls_pred = np.zeros(shape=num_images, dtype=np.int)

    i = 0
    while i < num_images:
        j = min(i + batch_size_test, num_images)
        feed_dict = {x: transfer_values_test[i:j],
                     y_true: test_labels[i:j]}
        cls_pred[i:j] = sess.run(y_pred_cls, feed_dict=feed_dict)
        i = j
    correct = (test_cls == cls_pred)
    print('Testing accuracy:', correct.mean())

training_step(101)
test_accuracy()

plt.plot(loss_graph, 'k-')
plt.title('Loss grafiği')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.show()