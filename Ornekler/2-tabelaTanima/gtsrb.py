import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from skimage import io
from skimage import transform
from skimage import exposure

import numpy as np

import random


data_path = 'D:/Isler/Yapay Zeka/GitHub/Ornekler/Data/gtsrb-german-traffic-sign/'

trainCsvPath = data_path + "Train.csv"
testCsvPath = data_path + "Test.csv"
signNamesCsvPath = data_path + "signnames.csv"

def get_class_names():
	labelNames = open(signNamesCsvPath).read().strip().split("\n")[1:]
	labelNames = [l.split(",")[1] for l in labelNames]
	return labelNames

def load_split(file_path):
	data=[]
	labels=[]
	rows = open(file_path).read().strip().split("\n")[1:]

	print(str(len(rows)) + " resim okunacak")

	random.shuffle(rows)
	for (i, row) in enumerate(rows):

		if i > 0 and i % 1000 == 0:
			print("preprocessed total %d images"%(i))

		(x1, y1, x2, y2, label, imagePath) = row.strip().split(",")[-6:]

		image = io.imread(data_path + imagePath)

		#w = int(x2) - int(x1)
		#h = int(y2) - int(y1)
		#bigSize = w if w > h else h

		image = image[int(y1):int(y2), int(x1):int(x2)]
		image = transform.resize(image, (32, 32))

		image = exposure.equalize_adapthist(image, clip_limit=0.1)

		data.append(image)
		labels.append(int(label))
		
	data = np.array(data)
	labels = np.array(labels)
	return (data, labels)

def get_data():
	cache_path = data_path + "Cache/"

	if os.path.isfile(cache_path+'trainX.npy') and os.path.isfile(cache_path+'trainY.npy'):
		print("Egitim datasi var.")
		trainX = np.load(cache_path + 'trainX.npy')
		trainY = np.load(cache_path + 'trainY.npy')
	else:
		print("Egitim datasi yok.")
		(trainX, trainY) = load_split(trainCsvPath)
		trainX = trainX.astype("float32") / 255.0

		np.save(cache_path + 'trainX.npy', trainX)
		np.save(cache_path + 'trainY.npy', trainY)

	print("Egitim datasi okundu.")

	if os.path.isfile(cache_path+'testX.npy') and os.path.isfile(cache_path+'testY.npy'):
		print("Test datasi var.")
		testX = np.load(cache_path + 'testX.npy')
		testY = np.load(cache_path + 'testY.npy')
	else:
		print("Test datasi yok.")
		(testX, testY) = load_split(testCsvPath)
		testX = testX.astype("float32") / 255.0

		np.save(cache_path + 'testX.npy', testX)
		np.save(cache_path + 'testY.npy', testY)

	print("Test datasi okundu.")

	return [trainX, trainY, testX, testY]