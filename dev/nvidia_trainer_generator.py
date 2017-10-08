#!/usr/bin/python

# other library imports
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Convolution2D, MaxPooling2D, Cropping2D, Dropout
from keras.backend import tf as ktf
'''
This file contains the method that holds the NVIDIA neural network model.
I modified the network architecture to have dropout after each layer.
This modification helped enormously in training the neural network.
The car stayed much more centered and did not fly off the road.
'''


def train_nvidia(train_generator, num_train_samples,
                 validation_generator, num_validation_samples,
                 nb_epoch=10, batch_size=32):
    """
    This method is used to get the model to then train.  The history object of the model is also returned.
    :param train_generator: The generator that fetches the next training data batch.
    :param num_train_samples: The number of training samples that the generator will eventually fetch.
    :param validation_generator: The generator that fetches the next validation data batch.
    :param num_validation_samples: The number of validation data that the validation generator will eventually fetch.
    :param nb_epoch: The number of epochs that the model will train for.
    :param batch_size: The batch size of each training mini run.
    :return: a tuple of 2 items containing (model, history object)
    """
    dropout_p = 0.5
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5,
                     input_shape=(160, 320, 3)))
    model.add(layer=Cropping2D(cropping=((70, 20), (0, 0))))
    # lenet architecture
    # Convolution #1
    model.add(layer=Convolution2D(nb_filter=24,
                                  nb_row=5,
                                  nb_col=5,
                                  subsample=(2, 2),
                                  activation='relu'))
    model.add(Dropout(dropout_p))
    # Convolution #2
    model.add(layer=Convolution2D(nb_filter=36,
                                  nb_row=5,
                                  nb_col=5,
                                  subsample=(2, 2),
                                  activation='relu'))
    model.add(Dropout(dropout_p))
    # Convolution #3
    model.add(layer=Convolution2D(nb_filter=48,
                                  nb_row=5,
                                  nb_col=5,
                                  subsample=(2, 2),
                                  activation='relu'))
    model.add(Dropout(dropout_p))
    # Convolution #4
    model.add(layer=Convolution2D(nb_filter=64,
                                  nb_row=3,
                                  nb_col=3,
                                  activation='relu'))
    model.add(Dropout(dropout_p))
    # Convolution #5
    model.add(layer=Convolution2D(nb_filter=64,
                                  nb_row=3,
                                  nb_col=3,
                                  activation='relu'))
    model.add(Dropout(dropout_p))
    # Flatten into fully connected layers
    model.add(layer=Flatten())
    model.add(layer=Dense(100))
    model.add(Dropout(dropout_p))
    model.add(layer=Dense(50))
    model.add(Dropout(dropout_p))
    model.add(layer=Dense(10))
    model.add(Dropout(dropout_p))
    model.add(layer=Dense(1))

    model.compile(loss='mse', optimizer='adam')
    history = model.fit_generator(generator=train_generator,
                                  samples_per_epoch=num_train_samples,
                                  validation_data=validation_generator,
                                  nb_val_samples=num_validation_samples,
                                  nb_epoch=nb_epoch)
    return model, history
