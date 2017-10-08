from dev_old.loader2 import load_data_from_files
from dev_old.nvidia_trainer import train_nvidia

file_names = [
#    '../data/driving_log.csv',
#     '../MyData1/driving_log.csv',
#     '../MyData2/driving_log.csv',
#    '../MyDataBridge1/driving_log.csv',
#    '../MyDataBridge3/driving_log.csv',
#     '../MyDataCorrection1/driving_log.csv',
    '../MyData3/driving_log.csv',
    '../MyData4/driving_log.csv',
    '../MyData5/driving_log.csv',
    '../MyData6/driving_log.csv'
]

#x_train, y_train = load_data(path_to_csv_file='../data/driving_log.csv')
x_train, y_train = load_data_from_files(file_names)

print("-data loaded-")

lenet_model, history = train_nvidia(x_train=x_train, y_train=y_train, nb_epoch=10)

lenet_model.save('../trained_models/lenet_model_3_cameras.h5')
