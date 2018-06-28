from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Dense
from keras.layers.core import Flatten
from keras.layers.core import Activation
from keras import backend as K

class LeNet:
    #model = conv2D relu maxpooling2d conv2d relu maxpooling2d flatten dense(500) relu dense(classes) softmax
    def build(height,width,depth,classes):
        model = Sequential()
        input_shape = (height,width,depth)

        if K.image_data_format() == "channels_first":
            input_shape = (depth,height,width)

        model.add(Conv2D(20,(5,5),padding="same",input_shape = input_shape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
        model.add(Conv2D(50,(5,5),padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        return model
