#!/usr/bin/python

# other library imports
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda


def train_reg(x_train, y_train):
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5,
                     input_shape=(160, 320, 3)))
    model.add(Flatten())
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    history = model.fit(x=x_train,
                        y=y_train,
                        validation_split=0.2,
                        shuffle=True)
    return model, history
