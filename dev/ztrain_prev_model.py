from dev.loader_generator import load_data_samples, generator
from sklearn.model_selection import train_test_split
from keras.models import load_model
from dev.nvidia_trainer_generator import train_nvidia

training_file_name = '../MyDatazTurnHardCorner03/driving_log.csv'
load_model_path = '../trained_models_sequence/zTrainSeq02.h5'
save_model_path = '../trained_models_sequence/zTrainSeq03.h5'

samples = load_data_samples(path_to_csv_file=training_file_name)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

model = load_model(load_model_path)

batch_size = 64
train_generator = generator(train_samples, batch_size=batch_size, side_cameras=False)
validation_generator = generator(validation_samples, batch_size=batch_size, side_cameras=False)


model.fit_generator(generator=train_generator,
                    samples_per_epoch=len(train_samples),
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_samples),
                    nb_epoch=10)

model.save(save_model_path)
