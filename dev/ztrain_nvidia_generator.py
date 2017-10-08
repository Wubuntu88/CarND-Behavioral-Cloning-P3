from dev.loader_generator import load_data_samples, generator
from sklearn.model_selection import train_test_split
from dev.nvidia_trainer_generator import train_nvidia

'''
This file performs the training of the neural network.
It is where the data locations are loaded, the training generator functions are created,
and where the model is trained and then saved.
'''

file_name = '../zAggregateData/AllDataLocations/all_data.csv'

samples = load_data_samples(path_to_csv_file=file_name)

print("-data loaded-")

train_samples, validation_samples = train_test_split(samples, test_size=0.4)

batch_size = 128

train_generator = generator(train_samples, batch_size=batch_size, side_cameras=False)
validation_generator = generator(validation_samples, batch_size=batch_size, side_cameras=False)

num_training_samples = len(train_samples) * 4
num_validation_samples = len(validation_samples) * 4

lenet_model, history = train_nvidia(train_generator=train_generator,
                                    num_train_samples=num_training_samples,
                                    validation_generator=validation_generator,
                                    num_validation_samples=num_validation_samples,
                                    nb_epoch=4,
                                    batch_size=batch_size)

lenet_model.save('../trained_models/nvidia_model_new_model.h5')
