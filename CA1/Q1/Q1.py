import numpy as np
import itertools


class MPNeuron():

    def __init__(self, weight):
        self.b = 2
        self.weights = weight

    def model(self, x):
        if self.weights @ x >= self.b:
            return 1
        else:
            return 0


# designing the network
def FA(x, y):
    neur1 = MPNeuron([2, -1])
    neur2 = MPNeuron([-1, 2])
    neur3 = MPNeuron([2, 2])
    neur4 = MPNeuron([1, 1])
    neur5 = MPNeuron([2, -1])
    neur6 = MPNeuron([-1, 2])
    neur7 = MPNeuron([2, 2])
    neur8 = MPNeuron([2, -1])
    neur9 = MPNeuron([-1, 2])
    neur10 = MPNeuron([2, 2])
    neur11 = MPNeuron([1, 1])
    neur12 = MPNeuron([1, 1])
    neur13 = MPNeuron([2, 2])
    # calculating the result for inputs
    z1 = neur1.model(np.array([x[1], y[1]]))
    z2 = neur2.model(np.array([x[1], y[1]]))
    z3 = neur3.model(np.array([z1, z2]))
    z4 = neur4.model(np.array([x[1], y[1]]))
    z5 = neur5.model(np.array([x[0], y[0]]))
    z6 = neur6.model(np.array([x[0], y[0]]))
    z7 = neur7.model(np.array([z5, z6]))
    z8 = neur8.model(np.array([z7, z4]))
    z9 = neur9.model(np.array([z7, z4]))
    z10 = neur10.model(np.array([z9, z8]))
    z11 = neur11.model(np.array([z4, z7]))
    z12 = neur12.model(np.array([x[0], y[0]]))
    z13 = neur13.model(np.array([z11, z12]))
    # 3 bit output
    return str(z13) + str(z10) + str(z3)


# inputs
x = [1, 0]
y = [1, 0]
c = list(itertools.product(x, y))
c1 = list(itertools.product(c, c))
for tup in c1:
    res = FA(tup[0], tup[1])
    print("The result of adding binary numbers", str(tup[0][0]) + str(tup[0][1]), "and",
          str(tup[1][0]) + str(tup[1][1]), "equals:", res)
