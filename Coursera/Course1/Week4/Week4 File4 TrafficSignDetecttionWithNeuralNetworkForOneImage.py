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

	img_size_width = 64
	img_size_height = 64
	base_data_path = "./Data/"	
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
	def get_resized_image(params):
		orginal_image = cv2.imread(params["path"], cv2.IMREAD_COLOR)
		new_image = cv2.resize(orginal_image, (params["width"], params["height"])) 

		write_log("Read: " + params["path"])

		return new_image

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

		params["educate_params"]["A0"] = params["A0"]
		params["educate_params"] = forward_propagation(params["educate_params"])

		L = len(params["educate_params"]["n"]) - 1
		return params["educate_params"]["A"+str(L)][0][0]

	def image_format(image):
		image = np.array([image])
		image = image.reshape(image.shape[0], -1).T
		image = image / 255.

		return image

	#Main
	if __name__ == '__main__':
		write_log("Starting...")

		if len(sys.argv) != 2:
			write_error("Incorrect Params! ex: python 4.py ./path/to/image.jpg")

		image = log_and_run(get_resized_image, "Read Image From Disk", {"path": sys.argv[1], "width": img_size_width, "height": img_size_height})
		image = log_and_run(image_format, "Format Image For Predict", image)
		
		parameters = log_and_run(read_educated_params, "Read Educated Params")
		parameters["m"] = image.shape[1]

		prediction = log_and_run(predict, "Predict for one image", {"educate_params": parameters, "A0": image})
		print("Image accuracy rate is " + str(prediction * 100) + "%")
		
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