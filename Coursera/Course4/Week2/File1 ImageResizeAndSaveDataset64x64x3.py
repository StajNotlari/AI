import numpy as np
#import matplotlib.pyplot as plt
import h5py
from scipy import ndimage
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
	resized_folder = "Resized"
	dataset_file = "dataset64x64x3.h5"
	dirs = { 
		"positive": "Train/Positive/",
		"negative": "Train/Negative/",
		"test": "Test/"
	}
	test_set_y = np.array([1, 1, 1, 0, 0, 1])

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

	#Image Resizing Functions
	def image_resize(source_image_path, destination_image_path, width, height):
		orginal_image = cv2.imread(source_image_path, cv2.IMREAD_COLOR)
		new_image = cv2.resize(orginal_image, (width, height)) 
		cv2.imwrite(destination_image_path, new_image)
		write_log("Resized: " + source_image_path + " -> " + destination_image_path)
		
		#np.array(ndimage.imread(destination_image_path, flatten=False))
		
		return new_image

	def create_folder_for_resized_image_if_not_exist(folder_path):
		if os.path.isdir(folder_path) == False:
			write_log("Create folder for resized images:" + folder_path)
			os.mkdir(folder_path, 777)

	def resize_all_image_in_folder(params):
		create_folder_for_resized_image_if_not_exist(params["destination"])

		images = []
		
		for img_name in os. listdir(params["source"]):
			if img_name.find(".") > -1:
				image = image_resize(params["source"]+img_name, params["destination"]+img_name, params["width"], params["height"])
				images.append(image)

		return np.array(images)

	def images_resize(params):
		global img_size_width, img_size_height, base_data_path

		rt = {}

		parameters = {
			"source": "",
			"destination": "",
			"width": img_size_width,
			"height": img_size_height
		}

		for name, path in params["dirs"].items():
			parameters["source"] = base_data_path + path
			parameters["destination"] = base_data_path + path + params["resized_folder"] + "/"
			rt[name] = log_and_run(resize_all_image_in_folder, path + " Images Resize", parameters)

		return rt

	def format_resized_images(resized_images):
		global test_set_y

		train_set_y = np.concatenate((np.zeros(resized_images["positive"].shape[0]) + 1, np.zeros(resized_images["negative"].shape[0])), axis=0)
		train_set_x = np.concatenate((resized_images["positive"], resized_images["negative"]), axis=0)
		
		test_set_x = resized_images["test"]

		return {
			"test_set_y": test_set_y,
			"test_set_x": test_set_x,
			"train_set_y": train_set_y,
			"train_set_x": train_set_x
		}
	
	def save_dataset(params):
		global base_data_path, dataset_file
		h5_file = h5py.File(base_data_path + dataset_file, 'w')

		for name, data in params.items():
			h5_file.create_dataset(name, data=data)

		h5_file.close()


	#Main
	if __name__ == '__main__':
		write_log("Starting...")

		params = {
			"dirs": dirs, 
			"resized_folder": resized_folder
		}
		resized_images = log_and_run(images_resize, "All Image Resize", params)

		dataset = log_and_run(format_resized_images, "Format Resized Images", resized_images) 
		
		log_and_run(save_dataset, "Save Dataset", dataset)

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