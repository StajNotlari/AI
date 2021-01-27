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
	verification_disable = False
	
	base_data_path = "./Data/"
	dataset_file = "dataset.h5"
	educated_params_file = "params.h5"

	learning_rate = 0.05
	keep_prop = 1
	lambd = 0.2

	#Log Settings
	logger = logging.getLogger(__name__)
	logger.setLevel(logging.INFO)

	handler = logging.FileHandler(log_file)
	handler.setLevel(logging.INFO)

	formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
	handler.setFormatter(formatter)

	logger.addHandler(handler)

	#Static random 
	np.random.seed(1)

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

	def intialize_deep_params_with_zeros(params):
		np.random.seed(2) 

		for l in range(1, len(params["n"])):
			params['W' + str(l)] = np.zeros((params["n"][l], params["n"][l - 1]))
			params['b' + str(l)] = np.zeros((params["n"][l], 1)) 

		log_and_run(params_verification, "Params Verification", params)

		return params

	def intialize_deep_params_with_random(params):
		np.random.seed(2) 

		for l in range(1, len(params["n"])):
			params['W' + str(l)] = np.random.randn(params["n"][l], params["n"][l - 1]) * 0.01
			params['b' + str(l)] = np.zeros((params["n"][l], 1)) 

		log_and_run(params_verification, "Params Verification", params)

		return params

	def intialize_deep_params_with_he(params):
		np.random.seed(2) 

		for l in range(1, len(params["n"])):
			params['W' + str(l)] = np.random.randn(params["n"][l], params["n"][l - 1]) * np.sqrt(2 / params["n"][l - 1])
			params['b' + str(l)] = np.zeros((params["n"][l], 1)) 

		log_and_run(params_verification, "Params Verification", params)

		return params

	def sigmoid(Z):
		return 1 / (1 + np.exp(-Z))

	def relu(Z):
		return np.maximum(0,Z)

	def param_is_numeric(name, param):
		try:
			assert(isinstance(param, float) or isinstance(param, int))
		except:
			print("Assert error for " + name + "!")
			print("Wanted: float, or integer")
			print("Got: " + type(param))
			sys.exit(0)
		

	def params_verification(params):
		global verification_disable
		if verification_disable:
			return

		numeric_params = {"m", "learning_rate", "num_iterations", "keep_prop", "lambd"}

		params_shapes = {
			"Y": (1, params["m"])
		}

		for l in range(1, len(params["n"])):
			param_is_numeric("n"+str(l), params["n"][l])

		params_shapes['A0'] = (params["n"][0], params["m"])

		for l in range(1, len(params["n"])):
			params_shapes['A' + str(l)] = (params["n"][l], params["m"])
			params_shapes['dA' + str(l)] = params_shapes['A' + str(l)]

			params_shapes['D' + str(l)] = (params["n"][l], params["m"])

			params_shapes['W' + str(l)] = (params["n"][l], params["n"][l-1])
			params_shapes['b' + str(l)] = (params["n"][l], 1)			
			params_shapes['dW' + str(l)] = params_shapes['W' + str(l)]
			params_shapes['db' + str(l)] = params_shapes['b' + str(l)]	

			params_shapes['Z' + str(l)] = (params["n"][l], params["m"])
			params_shapes['dZ' + str(l)] = params_shapes['Z' + str(l)]

		for name, param in params.items():
			if name == "costs" or name == "n":
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
		np.random.seed(5)
		for l in range(1, len(params["n"])):
			params["Z"+str(l)] = np.dot(params["W"+str(l)], params["A"+str(l-1)]) + params["b"+str(l)]
			if l == len(params["n"])-1:
				params["A"+str(l)] = sigmoid(params["Z"+str(l)])
			else:
				params["A"+str(l)] = relu(params["Z"+str(l)])

			if l < len(params["n"]) - 1:
				if params["keep_prop"] < 1:
					params["D"+str(l)] = np.random.rand(params["A"+str(l)].shape[0], params["A"+str(l)].shape[1])
					params["D"+str(l)] = params["D"+str(l)] < params["keep_prop"] 

					params["A"+str(l)] = params["A"+str(l)] * params["D"+str(l)]
					params["A"+str(l)] = params["A"+str(l)] / params["keep_prop"]   
		
		log_and_run(params_verification, "Params Verification", params)
		
		return params

	def calculate_cost(params):
		L = len(params["n"])-1

		cost = (-1 / params["m"]) * np.sum(np.multiply(params["Y"], np.log(params["A"+str(L)])) + np.multiply(1 - params["Y"], np.log(1 - params["A"+str(L)])))
		
		if params["lambd"] != 0:
			sum_W = 0
			for l in range(1, len(params["n"])):
				sum_W += np.sum(np.square(params["W"+str(l)]))
			cost += lambd * sum_W / (2 * params["m"])

		cost= np.squeeze(cost)

		assert(isinstance(cost, float))

		params["costs"].append(cost)
		
		log_and_run(params_verification, "Params Verification", params)
		
		return params

	def back_propagation(params):
		L = len(params["n"])-1
		
		params["dA"+str(L)] = - (np.divide(params["Y"], params["A"+str(L)]) - np.divide(1 - params["Y"], 1 - params["A"+str(L)]))
		
		for l in reversed(range(L)):
			l = l+1
			if l == L:
				s = 1/(1+np.exp(-params["Z"+str(l)]))
				params["dZ"+str(l)] = params["dA"+str(l)] * s * (1-s)
			else:
				params["dZ"+str(l)] = np.array(params["dA"+str(l)], copy=True) 
				params["dZ"+str(l)][params["Z"+str(l)] <= 0] = 0

			params["dW"+str(l)] = 1/params["m"] * np.dot(params["dZ"+str(l)], params["A"+str(l-1)].T)
			if params["lambd"] != 0 :
				lmb = (params["lambd"] * params["W"+str(l)]) / params["m"]
				params["dW"+str(l)] += lmb
			
			params["db"+str(l)] = 1/params["m"] * np.sum(params["dZ"+str(l)], axis = 1, keepdims = True)
			if l > 1:
				params["dA"+str(l-1)] = np.dot(params["W"+str(l)].T, params["dZ"+str(l)])

				if params["keep_prop"] < 1:
					params["dA"+str(l-1)] = params["dA"+str(l-1)] / params["keep_prop"]   

		log_and_run(params_verification, "Params Verification", params)

		return params

	def update_parameters(params):
		for l in range(1, len(params["n"])):
			params["W"+str(l)] -= params["learning_rate"] * params["dW"+str(l)]
			params["b"+str(l)] -= params["learning_rate"] * params["db"+str(l)]

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
		global verification_disable
		
		nx, m = params["A0"].shape

		assert(nx == params["educate_params"]["W1"].shape[1])

		Y_prediction = np.zeros((1, m))

		verification_disable = True
		params["educate_params"]["A0"] = params["A0"]
		params["educate_params"]["Y"] = params["Y"]
		params["educate_params"] = forward_propagation(params["educate_params"])
		verification_disable = False

		L = len(params["educate_params"]["n"])-1
		for i in range(params["educate_params"]["A"+str(L)].shape[1]):
			if params["educate_params"]["A"+str(L)][0,i] > 0.5:
				Y_prediction[0,i] = 1
			else:
				Y_prediction[0,i] = 0
		
		assert(Y_prediction.shape == (1, m))
		
		percent = 100 - np.mean(np.abs(Y_prediction - params["Y"])) * 100
		return [params["educate_params"]["Z"+str(L)], Y_prediction, percent]

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
			"m": train_set_x.shape[1],
			"A0": train_set_x,
			"Y": train_set_y,
			"costs": []
		}

		#Very bad
		"""parameters["n"] = []
		parameters["n"].append(train_set_x.shape[0]) 
		parameters["n"].append(20)
		parameters["n"].append(7)
		parameters["n"].append(5)		
		parameters["n"].append(train_set_y.shape[0])"""

		parameters["n"] = []
		parameters["n"].append(train_set_x.shape[0]) 
		parameters["n"].append(4)
		parameters["n"].append(3)		
		parameters["n"].append(train_set_y.shape[0])

		parameters["learning_rate"] = learning_rate
		parameters["lambd"] = lambd
		parameters["keep_prop"] = keep_prop

		parameters = log_and_run(intialize_deep_params_with_he, "Initialize W and b params", parameters)
		
		parameters["num_iterations"] = 501
		parameters = log_and_run(educate, "Educate", parameters)
		
		log_and_run(draw_costs_graphic, "Draw graphic", parameters)
		
		educated_params = { "n": parameters["n"] }
		for l in range(1, len(parameters["n"])):
			educated_params["W"+str(l)] = parameters["W"+str(l)]
			educated_params["b"+str(l)] = parameters["b"+str(l)]
			educated_params["Z"+str(l)] = parameters["Z"+str(l)]

		log_and_run(save_educated_params, "Save Educated Params", educated_params)
		
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