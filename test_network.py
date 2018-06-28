# python test_network.py --model ironman_not_ironman.model --input_img examples/ironman.jpg
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import imutils
import numpy as np
import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-m","--model",required=True,help = "path to output model")
ap.add_argument("-i","--input_img",required=True, help="path to input image")
args = vars(ap.parse_args())

image = cv2.imread(args["input_img"])
original_image = image.copy()

image = cv2.resize(image,(28,28))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

print('Loading our model..')
model = load_model(args["model"])

(notironman,ironman) = model.predict(image)[0]

label = "IronMan" if ironman > notironman else "Not IronMan"
proba = ironman if ironman > notironman else notironman
label = "{}: {:.2f}%".format(label, proba * 100)

# draw the label on the image
output = imutils.resize(original_image, width=400)
cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2)

# show the output image
cv2.imshow("Output", output)
cv2.waitKey(0)
