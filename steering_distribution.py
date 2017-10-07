
import matplotlib.pyplot as plt

with open('MyData4/driving_log.csv') as f:
    steering_list = []
    f.readline()
    for line in f:
        comps = line.split(",")
        steering_list.append(float(comps[3]))
    plt.hist(steering_list, bins=50)
    plt.show()
