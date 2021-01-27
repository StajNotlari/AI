import numpy as np
import matplotlib.pyplot as plt
import h5py
import tensorflow as tf
from tensorflow import keras
#import scipy
#from PIL import Image
#from scipy import ndimage
import cv2
import os  
import logging
import sys
import math
#from lr_utils import load_dataset

log_file = "./.log"

try:
	#Variables
	debug = True
	verbose = True
	
	base_data_path = "./Data/"
	dataset_file = "dataset.h5"
	model_file = "model.h5"

	epochs = 50

	#Log Settings
	logger = logging.getLogger(__name__)
	logger.setLevel(logging.INFO)

	handler = logging.FileHandler(log_file)
	handler.setLevel(logging.INFO)

	formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
	handler.setFormatter(formatter)

	logger.addHandler(handler)

	#Default Functions
	def write_log(message):
		if debug:
			logger.info('%s', message)
		if verbose:
			print("LOG -> " + message)
			
	def write_error(error):
		logger.error('%s', error)
		print("Error: " + error)
		sys.exit(0)
		
	def log_and_run(function, message, paramaters = None):
		write_log('"' + message + '" running...')

		if type(paramaters) == type(None):
			r = function()
		else:
			r = function(paramaters)
			
		write_log('"' + message + '" ok')
		return r

	#Deep Learning Functions
	def read_dataset():
		global base_data_path, dataset_file
		h5_file = h5py.File(base_data_path + dataset_file, 'r')

		train_set_x = np.array(h5_file["train_set_x"][:]) 
		train_set_y = np.array(h5_file["train_set_y"][:]) 
		test_set_x = np.array(h5_file["test_set_x"][:]) 
		test_set_y = np.array(h5_file["test_set_y"][:]) 

		h5_file.close()

		return train_set_x, train_set_y, test_set_x, test_set_y

	def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
		m = X.shape[0]               
		mini_batches = []
		np.random.seed(seed)
		
		permutation = list(np.random.permutation(m))
		shuffled_X = X[permutation,:,:,:]
		shuffled_Y = Y[permutation,:]

		num_complete_minibatches = math.floor(m/mini_batch_size) 
		for k in range(0, num_complete_minibatches):
			mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
			mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
			mini_batch = (mini_batch_X, mini_batch_Y)
			mini_batches.append(mini_batch)
		
		if m % mini_batch_size != 0:
			mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
			mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
			mini_batch = (mini_batch_X, mini_batch_Y)
			mini_batches.append(mini_batch)
		
		return mini_batches

	def create_placeholders(n_H0, n_W0, n_C0, n_y):
		X = tf.placeholder(shape=[None, n_H0, n_W0, n_C0], dtype=tf.float32)
		Y = tf.placeholder(shape=[None, n_y], dtype=tf.float32)
		
		return X, Y

	def initialize_parameters():
		tf.set_random_seed(1)
		
		W1 = tf.get_variable("W1", [4, 4, 3, 8], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
		W2 = tf.get_variable("W2", [2, 2, 8, 16], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
		
		parameters = {"W1": W1, "W2": W2}
		
		return parameters

	def forward_propagation(X, parameters):
		W1 = parameters['W1']
		W2 = parameters['W2']
		
		Z1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
		A1 = tf.nn.relu(Z1)
		P1 = tf.nn.max_pool(A1, ksize = [1, 8, 8, 1], strides = [1, 8, 8, 1], padding='SAME')
		
		Z2 = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding='SAME')
		A2 = tf.nn.relu(Z2)
		P2 = tf.nn.max_pool(A2, ksize = [1, 4, 4, 1], strides = [1, 4, 4, 1], padding='SAME')
		P2 = tf.contrib.layers.flatten(P2)
		
		Z3 = tf.contrib.layers.fully_connected(P2, 2, activation_fn=None)
		
		return Z3

	def compute_cost(Z3, Y):
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=Y))
		
		return cost

	def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.009, num_epochs = 5, minibatch_size = 64, print_cost = True):
		tf.set_random_seed(1)                             
		seed = 3                                          
		(m, n_H0, n_W0, n_C0) = X_train.shape             
		n_y = Y_train.shape[1]      
		                
		costs = []                                        
		
		X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)
		
		parameters = initialize_parameters()
		
		Z3 = forward_propagation(X, parameters)
		cost = compute_cost(Z3, Y)
		optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
		
		init = tf.global_variables_initializer()
		
		sess = tf.Session()

		sess.run(init)
		for epoch in range(num_epochs):

			minibatch_cost = 0.
			num_minibatches = int(m / minibatch_size)
			seed = seed + 1
			minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
			
			for minibatch in minibatches:
				(minibatch_X, minibatch_Y) = minibatch
				
				_ , temp_cost = sess.run([optimizer, cost], feed_dict={X:minibatch_X, Y:minibatch_Y})
				
				minibatch_cost += temp_cost / num_minibatches

			if print_cost == True and epoch % 5 == 0:
				print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
			if print_cost == True and epoch % 1 == 0:
				costs.append(minibatch_cost)

		
		with sess.as_default():
			predict_op = tf.argmax(Z3, 1)
			correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))

			accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

			print(accuracy)
			train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
			test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
			print("Train Accuracy:", train_accuracy)
			print("Test Accuracy:", test_accuracy)
				
		return costs, learning_rate

	def draw_costs_graph(costs, learning_rate):
		plt.plot(np.squeeze(costs))
		plt.ylabel('cost')
		plt.xlabel('iterations (per tens)')
		plt.title("Learning rate =" + str(learning_rate))
		plt.show()

	#Main
	if __name__ == '__main__':
		write_log("Starting...")

		train_set_x, train_set_y, test_set_x, test_set_y = log_and_run(read_dataset, "Read Dataset")

		train_set_y_sized = np.zeros([train_set_x.shape[0], 2])
		test_set_y_sized = np.zeros([test_set_x.shape[0], 2])

		i = 0
		for y in test_set_y:
			test_set_y_sized[i][0] = abs(y-1)
			test_set_y_sized[i][1] = y
			i += 1

		i = 0
		for y in train_set_y:
			train_set_y_sized[i][0] = abs(y-1)
			train_set_y_sized[i][1] = y
			i += 1
		
		costs, learning_rate = model(train_set_x, train_set_y_sized, test_set_x, test_set_y_sized)

		draw_costs_graph(costs, learning_rate)

		write_log("Success")

except SystemExit:
    print("")

except:
	try:
		def write_log(message):
			print(str(message))
			with open(log_file, "a") as f:
				f.write(str(message))
	except:
		print("write_log function defined")
    
	exc_type, exc_obj, exc_tb = sys.exc_info()    
	write_log("Error! -> " + str(", ".sys.exc_info()) + " (line: " + str(exc_tb.tb_lineno) + ")")