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
	debug = False
	verbose = False
	
	base_data_path = "./Data/"
	dataset_file = "dataset.h5"
	educated_params_file = "params.h5"

	learning_rate = 0.0005

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

	def intialize_params_with_zero(params):
		params["W"] = np.zeros([params["n"],1])
		params["b"] = 0
		
		log_and_run(params_verification, "Params Verification", params)
		
		return params

	def sigmoid(z):
		return 1 / (1 + np.exp(-z))

	def param_is_numeric(name, param):
		try:
			assert(isinstance(param, float) or isinstance(param, int))
		except:
			print("Assert error for " + name + "!")
			print("Wanted: float, or integer")
			print("Got: " + type(param))
			sys.exit(0)
		

	def params_verification(params):
		numeric_params = {"m", "n", "b", "learning_rate", "db", "num_iterations"}

		params_shapes = {
			"X": (params["n"], params["m"]),
			"Y": (1, params["m"]),
			"W": (params["n"], 1),
			"A": (1, params["m"]),
			"z": (1, params["m"]),
			"dz": (1, params["m"]),
			"dw": (params["n"], 1),
		}

		for name, param in params.items():
			if name == "costs":
				continue

			if name in numeric_params:
				param_is_numeric(name, param)
			else:
				try:
					assert(param.shape == params_shapes[name])
				except:
					print("Assert error for " + name + "!")
					print("Wanted: " + str(params_shapes[name]))
					print("Got: " + str(param.shape))
					sys.exit(0)
				

	def forward_propagation(params):
		
		params["z"] = np.dot(params["W"].T, params["X"]) + params["b"]
		params["A"] = sigmoid(params["z"])
		
		log_and_run(params_verification, "Params Verification", params)

		return params

	def calculate_cost(params):
		cost= - 1/params["m"] * np.sum((params["Y"] * np.log(params["A"]) + (1 - params["Y"]) * np.log(1 - params["A"])))
		cost= np.squeeze(cost)

		params["costs"].append(cost)
		
		log_and_run(params_verification, "Params Verification", params)
		
		return params

	def back_propagation(params):
		params["dz"] = params["A"] - params["Y"]
		params["dw"] = 1.0/params["m"] * np.dot(params["X"], params["dz"].T)
		params["db"] = 1.0/params["m"] * np.sum(params["dz"])

		log_and_run(params_verification, "Params Verification", params)
		
		return params

	def update_parameters(params):
		params["W"] -= params["learning_rate"] * params["dw"]
		params["b"] -= params["learning_rate"] * params["db"]

		log_and_run(params_verification, "Params Verification", params)
		
		return params

	def educate(params):
		for i in range(params["num_iterations"]):
			params = log_and_run(forward_propagation, "Forward Pagination", params)

			params = log_and_run(calculate_cost, "Calculate Cost", params)
			if i % 100 == 0:
				print("Cost is " + str(params["costs"][-1]) + " for " + str(i) + " iterations.")

			params = log_and_run(back_propagation, "Backward Pagination", params)			
			params = log_and_run(update_parameters, "Update Parameters", params)

		return params

	def draw_costs_graphic(params):
		costs = np.squeeze(params['costs'])
		plt.plot(costs)
		plt.ylabel('cost')
		plt.xlabel('iterations (per hundreds)')
		plt.title("Learning rate =" + str(params["learning_rate"]))
		plt.show()

	def predict(params):
		n, m = params["X"].shape

		assert(n == params["educate_params"]["W"].shape[0])

		Y_prediction = np.zeros((1, m))
		z = np.dot(params["educate_params"]["W"].T, params["X"]) + params["educate_params"]["b"]
		A = sigmoid(z)

		for i in range(A.shape[1]):
			if A[0,i] > 0.5:
				Y_prediction[0,i] = 1
			else:
				Y_prediction[0,i] = 0
		
		assert(Y_prediction.shape == (1, m))
		
		percent = 100 - np.mean(np.abs(Y_prediction - params["Y"])) * 100
		return [z, Y_prediction, percent]

	def save_educated_params(params):
		global base_data_path, educated_params_file
		h5_file = h5py.File(base_data_path + educated_params_file, 'w')

		for name, data in params.items():
			h5_file.create_dataset(name, data=data)

		h5_file.close()

	#Main
	if __name__ == '__main__':
		write_log("Starting...")

		train_set_x, train_set_y, test_set_x, test_set_y = log_and_run(read_dataset, "Read Dataset")

		parameters = {
			"X": train_set_x,
			"Y": train_set_y,
			"costs": []
		}

		parameters["n"], parameters["m"] = train_set_x.shape
		parameters["learning_rate"] = learning_rate
		
		parameters = log_and_run(intialize_params_with_zero, "Initialize W and b params", parameters)

		parameters["num_iterations"] = 10001
		parameters = log_and_run(educate, "Educate", parameters)
		

		log_and_run(draw_costs_graphic, "Draw graphic", parameters)

		educated_params = {
			"W": parameters["W"],
			"b": parameters["b"]
		}

		log_and_run(save_educated_params, "Save Educated Params", educated_params)
		
		train_set_prediction = log_and_run(predict, "Predict for Train", {"educate_params": parameters, "X": train_set_x, "Y": train_set_y})
		print("Train set accuracy rate is " + str(train_set_prediction[2]) + "%")

		test_set_prediction = log_and_run(predict, "Predict for Test ", {"educate_params": parameters, "X": test_set_x, "Y": test_set_y})
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