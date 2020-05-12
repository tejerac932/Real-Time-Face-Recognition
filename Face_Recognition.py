# SCRIPT
# python Face_Recognition.py -d face_detection_model -m openface_nn4.small2.v1.t7 -r output/recognizer.pickle -l output/label.pickle

###MUST BE USING LINUX MACHINE OR PYTHON 3.6 and below

from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os
import sys
import pathlib
import shutil

# Argument
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detector", required=True,
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--features-model", required=True,
	help="path to OpenCV's deep learning face features model")
ap.add_argument("-r", "--recognizer", required=True,
	help="path to model trained to recognize faces")
ap.add_argument("-l", "--label", required=True,
	help="path to label encoder")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

#Load Face Detector
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
#Load Feature model
feature_var = cv2.dnn.readNetFromTorch(args["features_model"])
#load Face recognition model
recognizer = pickle.loads(open(args["recognizer"], "rb").read())
le = pickle.loads(open(args["label"], "rb").read())
# initialize video stream
vs = VideoStream(src=0).start()
time.sleep(2.0)
# FPS estimator
fps = FPS().start()
# loop over frames from the video file stream
while True:
	#grab framw
	frame = vs.read()
	# resize the frame and grab dimensions
	frame = imutils.resize(frame, width=600)
	(h, w) = frame.shape[:2]
	# create a blob for frame
	frame_blob = cv2.dnn.blobFromImage(
		cv2.resize(frame, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)
	# detect the face from blob comparison
	detector.setInput(frame_blob)
	detections = detector.forward()
	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		# Check if face was found
		if confidence > args["confidence"]:
			# compute coordinates for bounding box
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			#isolate face
			face = frame[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]
			#find out if face is viewable
			if fW < 20 or fH < 20:
				continue
			# construct a blob and create a feature vector
			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
				(96, 96), (0, 0, 0), swapRB=True, crop=False)
			feature_var.setInput(faceBlob)
			vec = feature_var.forward()
			# Use face recognition to find the percentage of match to a class
			preds = recognizer.predict_proba(vec)[0]
			j = np.argmax(preds)
			proba = preds[j]
			name = le.classes_[j]
			# draw bounding box and probability of match
			text = "{}: {:.2f}%".format(name, proba * 100)
			y = startY - 10 if startY - 10 > 10 else startY + 10
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				(0, 0, 255), 2)
			cv2.putText(frame, text, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
	# update the FPS counter
	fps.update()
	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
# display video info
fps.stop()
print("elasped time: {:.2f}".format(fps.elapsed()))
print("approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()