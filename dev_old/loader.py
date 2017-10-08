#!/usr/bin/python
import csv
import cv2
import numpy as np


def load_data(path_to_csv_file: str) -> tuple:
    """

    :rtype: (np.array, np.array)
    """
    lines = []
    with open(path_to_csv_file) as csv_file:
        reader = csv.reader(csv_file)
        next(reader)
        for line in reader:
            lines.append(line)

    images = []
    steering_measurements = []

    for line in lines:
        steering_center = float(line[3])
        # create adjusted steering measurements for the side camera images
        correction = 0.2  # this is a parameter to tune
        steering_left = steering_center + correction
        steering_right = steering_center - correction

        for i in range(3):
            source_path = line[i]
            comps = source_path.split('/')
            filename = comps[-1]
            top_data_dir = path_to_csv_file.split('/')[-2]
            current_path = '../' + top_data_dir + '/IMG/' + filename
            # append the images & steering measurements as they are to the arrays of images and steering measurements.
            image = cv2.imread(current_path)
            images.append(image)

            if i == 0:  # center image
                measurement = float(line[3])
            elif i == 1:  # left image
                measurement = float(line[3]) + steering_left
            elif i == 2:  # right image
                measurement = float(line[3]) - steering_right
            else:
                raise Exception("invalid number for center, left or right image")
            steering_measurements.append(measurement)

            # now we append the flipped images of the data with sign-flipped measurements to emulate the car circling
            # the other way, because the car goes around the track one way
            # (at least with the default data given by Udacity)
            if i == 0:
                flipped_image = cv2.flip(src=image, flipCode=1)  # flipCode of 1 is flipping along y axis
                images.append(flipped_image)
                flipped_measurement = measurement * -1
                steering_measurements.append(flipped_measurement)

    x_train = np.array(images)
    y_train = np.array(steering_measurements)

    return x_train, y_train


def load_data_from_files(paths_to_files: list) -> tuple:
    assert type(paths_to_files) == list
    x_train_list = []
    y_train_list = []
    for file_path in paths_to_files:
        x_train_subset, y_train_subset = load_data(path_to_csv_file=file_path)
        x_train_list.append(x_train_subset)
        y_train_list.append(y_train_subset)
    print("------")
    [print(x.shape) for x in x_train_list]
    x_train_all = np.concatenate(x_train_list, axis=0)
    y_train_all = np.concatenate(tuple(y_train_list), axis=0)
    return x_train_all, y_train_all
