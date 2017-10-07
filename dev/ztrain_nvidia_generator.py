from dev.loader_generator import load_data_samples, generator
from sklearn.model_selection import train_test_split
from dev.nvidia_trainer_generator import train_nvidia

file_name = '../zAggregateData/AllDataLocations/all_data.csv'

samples = load_data_samples(path_to_csv_file=file_name)

print("-data loaded-")


train_samples, validation_samples = train_test_split(samples, test_size=0.2)

batch_size = 1028

train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

lenet_model, history = train_nvidia(train_generator=train_generator,
                                    )
#
# lenet_model.save('../trained_models/lenet_model_3_cameras.h5')
