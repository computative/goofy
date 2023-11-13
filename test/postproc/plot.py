import numpy as np
import sys

data = []
with open(sys.argv[1], "r") as infile:
    for line in infile:
        lst = eval(line.replace("\n","").replace(";","").replace(" ", ","))
        data.append(lst)

training = []
testing = []

for i, lst in enumerate(data):
    if (i % 2 == 0):
        training.append(lst)
    else:
        testing.append(lst)

training = np.array(training).T
testing = np.array(testing).T
x = np.array([100,200,400,800,1600,3200,6400])[0:len(testing[0])]

import matplotlib.pyplot as plt

assert training.shape == testing.shape

i = 0
for tr, te in zip(training,testing):
    plt.clf()
    plt.plot(x,tr)
    plt.plot(x,te)
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("sample size")
    plt.ylabel("rmse")
    plt.savefig(f"{sys.argv[1]}_{i}.png")
    i += 1 
