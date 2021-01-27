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

	def educate(params):
		global model_file

		img_width, img_height = params["X"][0].shape

		model = keras.Sequential([
			keras.layers.Flatten(input_shape=(img_width, img_height)),
			keras.layers.Dense(128, activation="relu"),
			keras.layers.Dense(2, activation="softmax")
		])

		model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

		model.fit(params["X"], params["Y"], epochs=params["epochs"])

		model.save(model_file)

		return model

	def get_model(params):
		global model_file
		if os.path.isfile(model_file) == False:
			return log_and_run(educate, "Educate", params)

		return keras.models.load_model(model_file)

	def predict(params):
		predict = params["model"].predict(params["X"])
		percent = (1 - np.sum(np.abs(params["Y"] - np.argmax(predict, axis=1))) / params["X"].shape[0]) * 100
		return predict, percent

	#Main
	if __name__ == '__main__':
		write_log("Starting...")

		train_set_x, train_set_y, test_set_x, test_set_y = log_and_run(read_dataset, "Read Dataset")

		train_set_x = train_set_x / 255.0
		test_set_x = test_set_x / 255.0

		parameters = {
			"X": train_set_x,
			"Y": train_set_y,
			"epochs": epochs
		}

		parameters["model"] = log_and_run(get_model, "Get Model", parameters)

		parameters["X"] = test_set_x
		parameters["Y"] = test_set_y
		_, percent = log_and_run(predict, "Predict test set", parameters)
		print("Dogruluk: " + str(percent))

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