
import tensorflow as tf
from tensorflow import keras
from keras.regularizers import  l2
def AlexNet(input_shapes = (224,224,3) , num_classes = 1000):

    alexnet = keras.Sequential(
        [
            # Input size : (224,224,3)

            keras.layers.Conv2D(filters = 96 ,
                                kernel_size = (11,11),
                                strides = (4,4),
                                padding = "same" ,
                                kernel_regularizer =l2(0.01),
                                input_shape = input_shapes),
            keras.layers.ReLU(),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D(pool_size = (2,2)),
            #Output shape (28,28,96)


            keras.layers.Conv2D(filters = 256 ,
                                kernel_size = (5,5),
                                padding = "same"),
            keras.layers.ReLU(),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D(pool_size=(2, 2)),
            #Output Shape :(12,12,256)

            keras.layers.Conv2D(filters=384,
                                kernel_size=(5, 5),
                                padding="same"),
            keras.layers.ReLU(),
            keras.layers.BatchNormalization(),

            keras.layers.Conv2D(filters=256,
                                kernel_size=(4, 4),
                                padding="same"),
            keras.layers.ReLU(),
            keras.layers.BatchNormalization(),

            #Now we'll flatten our output (14,14,256) --> 14*14*256

            #This is our fully connected part and also this part has the most of parameters
            keras.layers.Flatten(),
            keras.layers.Dense(units = 2048 ,
                               activation = "relu"),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(units = 2048 ,
                               activation = "relu"),
            keras.layers.Dropout(0.5),

            #output part:
            keras.layers.Dense(units = num_classes)


        ]
    )

    alexnet.compile(optimizer = "rmsprop" , loss = keras.losses.SparseCategoricalCrossentropy())


    return alexnet


model = AlexNet()
print(model.summary())