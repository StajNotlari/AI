import numpy as np
import matplotlib.pyplot as plt
import h5py
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
	educated_params_file = "params.h5"

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
	def read_educated_params():
		global base_data_path, educated_params_file
		h5_file = h5py.File(base_data_path + educated_params_file, 'r')
		
		params = {}
		params["n"] = np.array(h5_file["n"])

		for l in range(1, len(params["n"])):
			params["W"+str(l)] = np.array(h5_file["W"+str(l)][:])			
			params["b"+str(l)] = np.array(h5_file["b"+str(l)])		
			params["Z"+str(l)] = np.array(h5_file["Z"+str(l)])

		h5_file.close()

		return params

	def read_dataset():
		global base_data_path, dataset_file
		h5_file = h5py.File(base_data_path + dataset_file, 'r')

		train_set_x = np.array(h5_file["train_set_x"][:]) 
		train_set_y = np.array(h5_file["train_set_y"][:]) 
		test_set_x = np.array(h5_file["test_set_x"][:]) 
		test_set_y = np.array(h5_file["test_set_y"][:]) 

		h5_file.close()

		return train_set_x, train_set_y, test_set_x, test_set_y

	def sigmoid(z):
		return 1 / (1 + np.exp(-z))
	
	def relu(Z):
		return np.maximum(0,Z)

	def forward_propagation(params):
		for l in range(1, len(params["n"])):
			params["Z"+str(l)] = np.dot(params["W"+str(l)], params["A"+str(l-1)]) + params["b"+str(l)]
			if l == len(params["n"])-1:
				params["A"+str(l)] = sigmoid(params["Z"+str(l)])
			else:
				params["A"+str(l)] = relu(params["Z"+str(l)])
				
		return params

	def predict(params):
		nx, m = params["A0"].shape

		assert(nx == params["educate_params"]["W1"].shape[1])

		Y_prediction = np.zeros((1, m))

		params["educate_params"]["A0"] = params["A0"]
		params["educate_params"]["Y"] = params["Y"]
		params["educate_params"] = forward_propagation(params["educate_params"])

		L = len(params["educate_params"]["n"])-1
		for i in range(params["educate_params"]["A"+str(L)].shape[1]):
			if params["educate_params"]["A"+str(L)][0,i] > 0.5:
				Y_prediction[0,i] = 1
			else:
				Y_prediction[0,i] = 0
		
		assert(Y_prediction.shape == (1, m))
		
		percent = 100 - np.mean(np.abs(Y_prediction - params["Y"])) * 100
		return [params["educate_params"]["Z"+str(L)], Y_prediction, percent]

	#Main
	if __name__ == '__main__':
		write_log("Starting...")

		parameters = { }

		train_set_x, train_set_y, test_set_x, test_set_y = log_and_run(read_dataset, "Read Dataset")
		
		parameters = log_and_run(read_educated_params, "Read Educated Params")
		parameters["m"] = train_set_x.shape[1]

		train_set_prediction = log_and_run(predict, "Predict for Train", {"educate_params": parameters, "A0": train_set_x, "Y": train_set_y})
		print("Train set accuracy rate is " + str(train_set_prediction[2]) + "%")

		test_set_prediction = log_and_run(predict, "Predict for Test ", {"educate_params": parameters, "A0": test_set_x, "Y": test_set_y})
		print("Test set accuracy rate is " + str(test_set_prediction[2]) + "%")

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