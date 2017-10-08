from dev_old.lenet_trainer import train_lenet
from dev_old.loader2 import load_data_from_files

file_names = [
#    '../data/driving_log.csv',
#    '../MyData1/driving_log.csv',
#    '../MyData2/driving_log.csv',
#     '../MyDataBridge1/driving_log.csv',
#     '../MyDataBridge3/driving_log.csv',
#     '../MyDataCorrection1/driving_log.csv',
#    '../MyData3/driving_log.csv',
#    '../MyData4/driving_log.csv',
    '../MyData5/driving_log.csv',
    '../MyData6/driving_log.csv'
]

#x_train, y_train = load_data(path_to_csv_file='../data/driving_log.csv')
x_train, y_train = load_data_from_files(file_names)

print("-data loaded-")

lenet_model, history = train_lenet(x_train=x_train, y_train=y_train, nb_epoch=30)

lenet_model.save('../trained_models/lenet_model_3_cameras.h5')
