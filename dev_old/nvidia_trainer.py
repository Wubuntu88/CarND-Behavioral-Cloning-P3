#!/usr/bin/python

# other library imports
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Convolution2D, MaxPooling2D, Cropping2D
from keras.backend import tf as ktf


def train_nvidia(x_train, y_train, nb_epoch=10):
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5,
                     input_shape=(160, 320, 3)))
    model.add(layer=Cropping2D(cropping=((70, 20), (0, 0))))
    # lenet architecture
    # Convolution #1
    model.add(layer=Convolution2D(nb_filter=24,
                                  nb_row=5,
                                  nb_col=5,
                                  subsample=(2,2),
                                  activation='relu'))

    # Convolution #2
    model.add(layer=Convolution2D(nb_filter=36,
                                  nb_row=5,
                                  nb_col=5,
                                  subsample=(2, 2),
                                  activation='relu'))

    # Convolution #3
    model.add(layer=Convolution2D(nb_filter=48,
                                  nb_row=5,
                                  nb_col=5,
                                  subsample=(2, 2),
                                  activation='relu'))

    # Convolution #4
    model.add(layer=Convolution2D(nb_filter=64,
                                  nb_row=3,
                                  nb_col=3,
                                  activation='relu'))

    # Convolution #5
    model.add(layer=Convolution2D(nb_filter=64,
                                  nb_row=3,
                                  nb_col=3,
                                  activation='relu'))

    # Flatten into fully connected layers
    model.add(layer=Flatten())
    model.add(layer=Dense(100))
    model.add(layer=Dense(50))
    model.add(layer=Dense(10))
    model.add(layer=Dense(1))

    model.compile(loss='mse', optimizer='adam')
    history = model.fit(x=x_train,
                        y=y_train,
                        validation_split=0.2,
                        shuffle=True,
                        batch_size=128,
                        nb_epoch=nb_epoch)
    return model, history
