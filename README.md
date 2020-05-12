# Master-s-Project
There are three main aspects to this project that should be executed in order.
In order to download the feature model visit
https://github.com/pyannote/pyannote-data/blob/master/openface.nn4.small2.v1.t7

This code works only on a linux machine or a windows machine with python 3.6 or below.

1. Feature Extraction
The code features.py is used to extract the features in our database and create a neural network. 
Script: python features.py -i training_set -f output/features.pickle -d face_detection_model -m openface_nn4.small2.v1.t7
2. Training:
The code Train_images uses deep learning in order to train an algorithm to recognize the images and images similar in the dataset.
Script: python train_images.py -f output/features.pickle -r output/recognizer.pickle -l output/label.pickle
3. Test the recognition
The code Face_Recognition starts a video in which face recognition is applied in real time. 
Script: python Face_Recognition.py -d face_detection_model -m openface_nn4.small2.v1.t7 -r output/recognizer.pickle -l output/label.pickle
