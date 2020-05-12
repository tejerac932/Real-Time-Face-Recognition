###SCRIPT
### python features.py -i training_set -f output/features.pickle -d face_detection_model -m openface_nn4.small2.v1.t7

###MUST BE USING LINUX MACHINE OR PYTHON 3.6 and below


###Import packages
import os
import cv2
import argparse
import numpy as np
import pickle
import imutils
from imutils import paths

### Arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--training_set", required=True,
	help="path to training set")
ap.add_argument("-f", "--features", required=True,
	help="path for output features")
ap.add_argument("-d", "--detector", required=True,
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--feature-model", required=True,
	help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-c", "--confidence", type=float, default=0.8,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())


###Initializations
#Load face detector
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
#Load the feature model and training set
feature = cv2.dnn.readNetFromTorch(args["feature_model"])
trainpath = list(paths.list_images(args["training_set"]))
knownfeatures = [] #list of feature vectors within the dataset
knownLabels = [] #list of the respective labels
total = 0 #total number of faces in the trainer

# loop training set images
for (i, path) in enumerate(trainpath):
	#Print the progress
	print("processing image {}/{}".format(i + 1,
		len(trainpath)))
	# identify the image based off of their name
	name = path.split(os.path.sep)[-2]
	# load the image, resize it, and grab dimensions
	image = cv2.imread(path)
	image = imutils.resize(image, width=600)
	(h, w) = image.shape[:2]
	# blob the image(necessary for face detection)
	imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(image, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)
	# use the detector to localize faces in the image
	detector.setInput(imageBlob)
	detections = detector.forward()
	#check if face was detected
	if len(detections) > 0:
		i = np.argmax(detections[0, 0, :, 2])
		confidence = detections[0, 0, i, 2]
		### check if the face detected met the minimum confidence
		if confidence > args["confidence"]:
			# compute coordinates for bounding box
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			# isolate the face
			face = image[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]
			if fW < 20 or fH < 20:
				continue
			# take a blob of the face, transform into a feature vector
			# and place into the neural network
			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
				(96, 96), (0, 0, 0), swapRB=True, crop=False)
			feature.setInput(faceBlob)
			vec = feature.forward()
			knownLabels.append(name)
			knownfeatures.append(vec.flatten())
			total += 1 #increment number of faces in the deep learning
# Output features into a pickle
data = {"features": knownfeatures, "names": knownLabels}
f = open(args["features"], "wb")
f.write(pickle.dumps(data))
f.close()