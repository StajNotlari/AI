import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

w = tf.Variable(0, dtype=tf.float32)

# w**2 - 10 * w + 25 = 0
cost = tf.math.add(tf.math.multiply(-10., w), tf.math.add(w**2, 25))

train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

session = tf.Session()
session.run(tf.global_variables_initializer())

print("w: " + str(session.run(w)))

iterations = 500

for i in range(iterations):
        session.run(train)

print("w: " + str(session.run(w)))