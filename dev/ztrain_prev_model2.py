from dev_old.loader2 import load_data
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

training_file_name = '../zAggregateData/AllDataLocations/all_data.csv'
# training_file_name = '..//zAggregateData/symlinksToDrivingLogs/first_hard_turn_all_logs.csv'
# load_model_path = '../trained_models_sequence/nvidia_model_new_model_pretty_good.h5'
load_model_path = '../trained_models_sequence/zTrainAlmostPerfect5.h5'
save_model_path = '../trained_models_sequence/zTrainRebirth.h5'

model = load_model(load_model_path)

x_train, y_train = load_data(path_to_csv_file=training_file_name)


model.fit(x=x_train,
          y=y_train,
          batch_size=128,
          nb_epoch=5)

model.save(save_model_path)
