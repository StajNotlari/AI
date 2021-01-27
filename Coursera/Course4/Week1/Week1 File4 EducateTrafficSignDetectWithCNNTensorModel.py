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
	dataset_file = "dataset32x32x3.h5"
	model_file = "model32x32x3.h5"

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

	def get_model():
		model = keras.Sequential()

		model.add(tf.layers.Conv2D(filters=6, kernel_size=(5, 5), activation='relu', input_shape=(32,32,3)))
		model.add(tf.layers.AveragePooling2D(pool_size=(14, 14), strides=2))
		
		model.add(tf.layers.Conv2D(filters=16, kernel_size=(5, 5), activation='relu'))		
		model.add(tf.layers.AveragePooling2D(pool_size=(3, 3), strides=2))
		
		model.add(tf.layers.Flatten())

		model.add(tf.layers.Dense(units=120, activation='relu'))

		model.add(tf.layers.Dense(units=84, activation='relu'))

		model.add(tf.layers.Dense(units=2, activation = 'softmax'))

		return model

	def educate():
		global model_file

		model = log_and_run(get_model, "Get Model")

		model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

		model.fit(train_set_x, train_set_y, epochs=100)
		
		model.save(model_file)

		return model


	def get_educated_model():
		global model_file
		if os.path.isfile(model_file) == False:
			return log_and_run(educate, "Educate")

		return keras.models.load_model(model_file)
		
	def prediction(params):
		predict = params["model"].predict(params["X"])
		percent = (1 - np.sum(np.abs(params["Y"] - np.argmax(predict, axis=1))) / params["X"].shape[0]) * 100
		return predict, percent

	#Main
	if __name__ == '__main__':
		write_log("Starting...")

		train_set_x, train_set_y, test_set_x, test_set_y = log_and_run(read_dataset, "Read Dataset")

		params = {
			"X": train_set_x,
			"Y": train_set_y
		}
		params["model"] = log_and_run(get_educated_model, "Get Educated Model")

		_, train_percent = log_and_run(prediction, "Train Set Prediction", params)

		params["X"] = test_set_x
		params["Y"] = test_set_y
		_, test_percent = log_and_run(prediction, "Test Set Prediction", params)


		print("Train set: " + str(train_percent))
		print("Test set: " + str(test_percent))

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