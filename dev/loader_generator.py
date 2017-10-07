#!/usr/bin/python
import csv
import cv2
import numpy as np
from sklearn.utils import shuffle
'''
In this file, there are generator functions that load the data on demand (not at once).
This allows devices with restricted memory the ability to train the network.
'''


def load_data_samples(path_to_csv_file: str) -> tuple:
    """
    Loads the data about where the locations of the picture files are, as well as the steering directions.
    :param path_to_csv_file: the path to where the csv file is located.
    :rtype: list of lists
        Each element in the list represents a line,
        and each value in the 'line' represents a item from the tab delimited file
    """
    lines = []
    with open(path_to_csv_file) as csv_file:
        reader = csv.reader(csv_file)
        next(reader)
        for line in reader:
            lines.append(line)
    return lines


def generator(samples, batch_size=32, side_cameras=True):
    """
    Generator function that generates the training images used in the neural network.
    Note: 4 or 2 times the amount of data is generated than what is declared in the batch size.
    This is because of data augmentation (images are flipped) (+1), and the use of side cameras (+2)
    :param samples: The training data (data locations) from which pictures will be loaded and then generated
    :param batch_size:
    :param side_cameras: If True, the side camera data will be used, if false, the side camera images are ignored.
    """
    num_samples = len(samples)

    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            steering_measurements = []

            for batch_sample in batch_samples:
                steering_center = float(batch_sample[3])

                source_path_center = batch_sample[0]
                #  append the images & steering measurements as they are to the arrays of images and steering measurements.
                image_center = cv2.imread(source_path_center)
                images.append(image_center)
                steering_measurements.append(steering_center)

                # now we append the flipped images of the data with sign-flipped measurements to emulate the car circling
                # the other way, because the car goes around the track one way
                # (at least with the default data given by Udacity)
                flipped_image = cv2.flip(src=image_center, flipCode=1)  # flipCode of 1 is flipping along y axis
                images.append(flipped_image)
                flipped_measurement = steering_center * -1
                steering_measurements.append(flipped_measurement)

                if side_cameras is True:
                    # for side cameras
                    # create adjusted steering measurements for the side camera images
                    correction = 0.2  # this is a parameter to tune
                    steering_left = steering_center + correction
                    steering_right = steering_center - correction

                    source_path_left = batch_sample[1]
                    source_path_right = batch_sample[2]

                    image_left = cv2.imread(source_path_left)
                    image_right = cv2.imread(source_path_right)

                    images.append(image_left)
                    images.append(image_right)

                    steering_measurements.append(min(1.0, steering_left))
                    steering_measurements.append(max(-1.0, steering_right))

            x_train = np.array(images)
            y_train = np.array(steering_measurements)

            yield x_train, y_train
