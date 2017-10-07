from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Convolution2D, MaxPooling2D, Cropping2D, Dropout

'''
This code makes a visualization of the neural network model and saves it in model.png

!!!!! PROGRAMMERS: YE BE WARNED !!!!!
This code will not work on the carnd-term1 anaconda environment.
That is because the environment uses an ancient version of keras,
and there are some issues with that version finding pydot, graphviz (or something like that)
If you want to execute this code, you will need to install a keras version of 2.0 or higher.
(I got keras 2.0.5, which was the newest version at that time.)
'''

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

plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
