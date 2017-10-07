import matplotlib.pyplot as plt

# change the path, and intermediate paths to choose the data you want to visualize
path = '../ledata/'
file = 'driving_log.csv'
files = [
    path + 'MyData6/' + file,
    path + 'MyData7/' + file,
    path + 'MyData8/' + file,
    path + 'MyData9/' + file,
]
measurements = [[] for x in range(len(files))]
i = 0
for f_name in files:
    with open(f_name) as f:
        f.readline()
        for line in f:
            comps = line.split(",")
            steering_value = float(comps[3])
            measurements[i].append(steering_value)
        i += 1
plt.hist(measurements, bins=20, normed=True)
plt.xlabel('Steering value\n(negative -> turning left; positive -> turning right)', fontsize=16)
plt.ylabel('count of steering value', fontsize=16)
plt.title('Normed Histogram of steering values in sample files'
          '\n(Comparison of distributions in multiple files)', fontsize=20)
plt.show()
