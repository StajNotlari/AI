import numpy as np
import matplotlib.pyplot as plt
import h5py
import math
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

	optimizer = "adam" #gd or momentum or adam

	mini_batch_size = 64
	num_epochs = 2001

	learning_rate = 0.05
	keep_prop = 1
	lambd = 0.2
	beta1 = 0.9
	beta2 = 0.999
	epsilon = 1e-8
	adam_counter = 0  

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

	def initialize_adam(params) :
		for l in range(len(params["n"])-1):
			l += 1
			params["vdW" + str(l)] = np.zeros(params["W" + str(l)].shape)
			params["vdb" + str(l)] = np.zeros(params["b" + str(l)].shape)
			params["sdW" + str(l)] = np.zeros(params["W" + str(l)].shape)
			params["sdb" + str(l)] = np.zeros(params["b" + str(l)].shape)
		
		log_and_run(params_verification, "Params Verification", params)

		return params

	def initialize_velocity(params):
		for l in range(len(params["n"])-1):
			l += 1
			params["vdW" + str(l)] = np.zeros(params["W" + str(l)].shape)
			params["vdb" + str(l)] = np.zeros(params["b" + str(l)].shape)

		log_and_run(params_verification, "Params Verification", params)

		return params

	def update_parameters_with_momentum(params):
		for l in range(len(params["n"])-1):
			l += 1
			params["vdW" + str(l)] = params["beta1"] * params["vdW" + str(l)] + (1 - params["beta1"]) * params["dW" + str(l)]
			params["vdb" + str(l)] = params["beta1"] * params["vdb" + str(l)] + (1 - params["beta1"]) * params["db" + str(l)]
			
			params["W" + str(l)] = params["W" + str(l)] - params["learning_rate"] * params["vdW" + str(l)]
			params["b" + str(l)] = params["b" + str(l)] - params["learning_rate"] * params["vdb" + str(l)]

		log_and_run(params_verification, "Params Verification", params)
		
		return params

	def update_parameters_with_adam(params):
		for l in range(len(params["n"])-1):
			l += 1

			params["vdW" + str(l)] = params["beta1"] * params["vdW" + str(l)] + (1 - params["beta1"]) * params['dW' + str(l)]
			params["vdb" + str(l)] = params["beta1"] * params["vdb" + str(l)] + (1 - params["beta1"]) * params['db' + str(l)]
			
			params["vdW_corrected" + str(l)] = params["vdW" + str(l)] / (1 - np.power(params["beta1"], params["adam_counter"]))
			params["vdb_corrected" + str(l)] = params["vdb" + str(l)] / (1 - np.power(params["beta1"], params["adam_counter"]))
			
			params["sdW" + str(l)] = params["beta2"] * params["sdW" + str(l)] + (1 - params["beta2"]) * np.power(params['dW' + str(l)], 2)
			params["sdb" + str(l)] = params["beta2"] * params["sdb" + str(l)] + (1 - params["beta2"]) * np.power(params['db' + str(l)], 2)
			
			
			params["sdW_corrected" + str(l)] = params["sdW" + str(l)] / (1 - np.power(params["beta2"], params["adam_counter"]))
			params["sdb_corrected" + str(l)] = params["sdb" + str(l)] / (1 - np.power(params["beta2"], params["adam_counter"]))
			
			params["W" + str(l)] = params["W" + str(l)] - params["learning_rate"] * params["vdW_corrected" + str(l)] / np.sqrt(params["sdW_corrected" + str(l)] + params["epsilon"])
			params["b" + str(l)] = params["b" + str(l)] - params["learning_rate"] * params["vdb_corrected" + str(l)] / np.sqrt(params["sdb_corrected" + str(l)] + params["epsilon"])
			
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
		
	def fill_current_mini_batch_size(params):
		params["current_mini_batch_size"] = params["mini_batch_size"]

		if "mini_batches" in params and "current_mini_batch_step" in params:
			if params["current_mini_batch_step"] == len(params["mini_batches"]["X"]) - 1:
				if params["m"] % params["mini_batch_size"] > 0:
					params["current_mini_batch_size"] = params["m"] % params["mini_batch_size"]

		return params

	def get_numeric_params():
		numeric_params = {
			"m",
			"learning_rate", 
			"beta1", 
			"beta2", 
			"epsilon", 
			"num_epochs",
			"mini_batch_size", 
			"num_iterations", 
			"adam_counter",
			"keep_prop", 
			"current_mini_batch_step",
			"current_mini_batch_size",
			"lambd"}
		return numeric_params

	def get_params_shapes(params):
		params_shapes = {
			"Y_orj": (1, params["m"]),
			"X_orj": (params["n"][0], params["m"]),
			"Y_shuffled": (1, params["m"]),
			"X_shuffled": (params["n"][0], params["m"])
		}

		params_shapes['A0'] = (params["n"][0], params["current_mini_batch_size"])
		params_shapes['Y'] = (1, params["current_mini_batch_size"])

		for l in range(1, len(params["n"])):
			params_shapes['A' + str(l)] = (params["n"][l], params["current_mini_batch_size"])
			params_shapes['dA' + str(l)] = params_shapes['A' + str(l)]

			params_shapes['D' + str(l)] = (params["n"][l], params["m"])

			params_shapes['W' + str(l)] = (params["n"][l], params["n"][l-1])
			params_shapes['b' + str(l)] = (params["n"][l], 1)			
			params_shapes['dW' + str(l)] = params_shapes['W' + str(l)]
			params_shapes['db' + str(l)] = params_shapes['b' + str(l)]	
			params_shapes['vdW' + str(l)] = params_shapes['W' + str(l)]
			params_shapes['vdb' + str(l)] = params_shapes['b' + str(l)]	
			params_shapes['sdW' + str(l)] = params_shapes['W' + str(l)]
			params_shapes['sdb' + str(l)] = params_shapes['b' + str(l)]
			params_shapes['vdW_corrected' + str(l)] = params_shapes['W' + str(l)]
			params_shapes['vdb_corrected' + str(l)] = params_shapes['b' + str(l)]	
			params_shapes['sdW_corrected' + str(l)] = params_shapes['W' + str(l)]
			params_shapes['sdb_corrected' + str(l)] = params_shapes['b' + str(l)]	

			params_shapes['Z' + str(l)] = (params["n"][l], params["current_mini_batch_size"])
			params_shapes['dZ' + str(l)] = params_shapes['Z' + str(l)]

		return params_shapes

	def params_verification(params):
		global verification_disable
		if verification_disable:
			return

		numeric_params = get_numeric_params()
		params = fill_current_mini_batch_size(params)
		params_shapes = get_params_shapes(params)

		for l in range(1, len(params["n"])):
			param_is_numeric("n"+str(l), params["n"][l])
		
		no_validate = ["costs", "n", "mini_batches"]

		for name, param in params.items():
			if name in no_validate:
				continue

			if name in numeric_params:
				param_is_numeric(name, param)
			else:
				try:
					assert(param.shape == params_shapes[name])
				except:	
					if name[0:2] == "dA" or name[0:2] == "dZ":
						return
										
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

	def fill_random_mini_batches(params):
		permutation = list(np.random.permutation(389))
		params["X_shuffled"] = params["X_orj"][:, permutation]
		params["Y_shuffled"] = params["Y_orj"][:, permutation]
		
		params["mini_batches"] = {
			"X": [],
			"Y": []
		}

		i = math.floor(params["m"]/params["mini_batch_size"])
		for k in range(0, int(i) + 1):
			mini_batch_X = params["X_shuffled"][:,k * params["mini_batch_size"]:(k + 1) * params["mini_batch_size"]]
			mini_batch_Y = params["Y_shuffled"][:,k * params["mini_batch_size"]:(k + 1) * params["mini_batch_size"]]
			
			params["mini_batches"]["X"].append(mini_batch_X)			
			params["mini_batches"]["Y"].append(mini_batch_Y)

		return params

	def educate(params):
		params = log_and_run(fill_random_mini_batches, "Fill mini batches", params)
		
		for i in range(params["num_epochs"]):
			for k in range(len(params["mini_batches"]["X"])):
				params["current_mini_batch_step"] = k

				params["A0"] = params["mini_batches"]["X"][k]
				params["Y"] = params["mini_batches"]["Y"][k]

				params = log_and_run(forward_propagation, "Forward Pagination", params)

				params = log_and_run(calculate_cost, "Calculate Cost", params)

				params = log_and_run(back_propagation, "Backward Pagination", params)

				if optimizer == "gd":
					update_function = update_parameters
				elif optimizer == "momentum":
					update_function = update_parameters_with_momentum
				elif optimizer == "adam":
					params["adam_counter"] += 1
					update_function = update_parameters_with_adam

				params = log_and_run(update_function, "Update Parameters", params)

			if i % 100 == 0:
				print("Cost is " + str(params["costs"][-1]) + " for " + str(i) + " epochs.")

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
			"mini_batch_size": mini_batch_size,
			"m": train_set_x.shape[1],
			"X_orj": train_set_x,
			"Y_orj": train_set_y,
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
		parameters["beta1"] = beta1
		parameters["beta2"] = beta2
		parameters["epsilon"] = epsilon
		parameters["num_epochs"] = num_epochs
		parameters["adam_counter"] = adam_counter


		parameters = log_and_run(intialize_deep_params_with_he, "Initialize W and b params", parameters)
		if optimizer == "momentum":
			parameters = initialize_velocity(parameters)
		elif optimizer == "adam":
			parameters = initialize_adam(parameters)
		
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