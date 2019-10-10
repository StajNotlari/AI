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

		W = np.array(h5_file["W"][:]) 
		b = np.array(h5_file["b"]) 

		h5_file.close()

		return W, b

	def sigmoid(z):
		return 1 / (1 + np.exp(-z))

	def predict(params):
		n, m = params["X"].shape

		assert(n == params["educate_params"]["W"].shape[0])

		z = np.dot(params["educate_params"]["W"].T, params["X"]) + params["educate_params"]["b"]
		A = sigmoid(z)

		return A[0][0]

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
		
		parameters = { }
		parameters["n"], parameters["m"] = image.shape		
		parameters["W"], parameters["b"] = log_and_run(read_educated_params, "Read Educated Params")
		
		prediction = log_and_run(predict, "Predict for one image", {"educate_params": parameters, "X": image})
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