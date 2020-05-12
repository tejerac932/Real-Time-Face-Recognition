###SCRIPT
# python train_images.py -f output/features.pickle -r output/recognizer.pickle -l output/label.pickle

###MUST BE USING LINUX MACHINE OR PYTHON 3.6 and below


from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import argparse
import pickle

### Argument
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--features", required=True,
	help="path to serialized db of facial features")
ap.add_argument("-r", "--recognizer", required=True,
	help="path to output model trained to recognize faces")
ap.add_argument("-l", "--label", required=True,
	help="path to output label encoder")
args = vars(ap.parse_args())

# load features
data = pickle.loads(open(args["features"], "rb").read())
# find the labels
le = LabelEncoder()
labels = le.fit_transform(data["names"])
# apply deep learning on the set of feature vectors
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["features"], labels)
# create a face recognizer model(used for feature comparison)
f = open(args["recognizer"], "wb")
f.write(pickle.dumps(recognizer))
f.close()
# output the labels
f = open(args["label"], "wb")
f.write(pickle.dumps(le))
f.close()