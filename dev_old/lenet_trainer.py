#!/usr/bin/python

# other library imports
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Convolution2D, MaxPooling2D, Cropping2D
from keras.backend import tf as ktf


def train_lenet(x_train, y_train, nb_epoch=10):
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5,
                     input_shape=(160, 320, 3)))
    model.add(layer=Cropping2D(cropping=((70, 20), (0, 0))))
    # lenet architecture
    # Convolution #1
    model.add(layer=Convolution2D(nb_filter=6,
                                  nb_row=5,
                                  nb_col=5,
                                  activation='relu'))
    # Pooling #1
    model.add(layer=MaxPooling2D())

    # Convolution #2
    model.add(layer=Convolution2D(nb_filter=6,
                                  nb_row=5,
                                  nb_col=5,
                                  activation='relu'))
    # Pooling #2
    model.add(layer=MaxPooling2D())

    # Flatten into fully connected layers
    model.add(layer=Flatten())
    model.add(layer=Dense(120))
    model.add(layer=Dense(84))
    model.add(layer=Dense(1))

    model.compile(loss='mse', optimizer='adam')
    history = model.fit(x=x_train,
                        y=y_train,
                        validation_split=0.2,
                        shuffle=True,
                        batch_size=256,
                        nb_epoch=nb_epoch)
    return model, history
