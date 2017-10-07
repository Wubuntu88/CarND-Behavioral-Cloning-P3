import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("whitegrid")
filename = '../ledata/MyData9/driving_log.csv'  # change the filenate to your desired file
with open(filename) as f:
    steering_list = []
    f.readline()
    for line in f:
        comps = line.split(",")
        steering_list.append(float(comps[3]))
    plt.hist(steering_list, bins=20)
    plt.xlabel('Steering value\n(negative -> turning left; positive -> turning right)', fontsize=16)
    plt.ylabel('count of steering value', fontsize=16)
    plt.title('Histogram of steering values in sample file', fontsize=20)
    plt.show()
