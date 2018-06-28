# python train_network.py --dataset images --model ironman_not_ironman.model
import matplotlib
matplotlib.use("Agg")   #saving backend to Agg so that we can save plot to disk in background

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from imageclassifymodel.lenet import LeNet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import random
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d","--dataset",required=True,help="path to input dataset")
ap.add_argument("-m","--model",required=True,help = "path to output model")
ap.add_argument("-p","--plot",type=str,default="plot.png", help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

EPOCHS = 25
BS = 32
INIT_LR = 1e-3

data = []
labels = []

#dataset has path of input images. grab it and form a list. randomize it
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(10)
random.shuffle(imagePaths)

#take each path from imagePaths i.e load each image from the list of paths of images

#convert each img to size 28X28 and convert to an array for feeding to NN
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    image = cv2.resize(image,(28,28))
    image = img_to_array(image)
    data.append(image)

    label = imagePath.split(os.path.sep)[-2]
    label = 1 if label == "ironman" else 0
    labels.append(label)

data = np.array(data,dtype = "float")/255.0
labels = np.array(labels)

trainX,testX,trainY,testY = train_test_split(data,labels,test_size=0.25,random_state=42)

trainY = to_categorical(trainY,num_classes=2)
testY = to_categorical(testY,num_classes=2)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

print("compiling model..")
model = LeNet.build(height=28,width=28,depth=3,classes=2)
opt = Adam(lr=INIT_LR,decay=INIT_LR/EPOCHS)
model.compile(optimizer=opt,loss="binary_crossentropy",metrics=["accuracy"])

print("Training network..")
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=1)

print("Serializing model..")
model.save(args["model"])

plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on IronMan/Not IronMan")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])

