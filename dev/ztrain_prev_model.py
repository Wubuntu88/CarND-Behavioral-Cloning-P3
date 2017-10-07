from dev.loader_generator import load_data_samples, generator
from sklearn.model_selection import train_test_split
from keras.models import load_model

'''
In this file, I load a previously trained network, and train it on new data.
The benefit of this is that we can choose a model that turned out well, and then refine it.
I refine it by noticing the problem spots (where the car drives off the road),
and then collect data of myself driving that section of road, but with the desired behavior.
I then load the model, load the data from that specific run, and train the model only on that data.
This training fixes some of the problems that it was previously having, although sometimes it seems that the
model learns frustratingly slow.  One thing to note is that this kind of training can also make the model
perform worse on certain sections of the road it had previously been good at.
However, my experience is that it certainly gives more benefits than consequences.
'''

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
