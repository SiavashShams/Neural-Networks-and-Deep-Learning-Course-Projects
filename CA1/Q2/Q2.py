import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

x1 = np.random.normal(0, 0.3, 100)
y1 = np.random.normal(0, 0.3, 100)
x2 = np.random.normal(2, 0.3, 100)
y2 = np.random.normal(2, 0.3, 100)
# uncomment to see next part
'''
x1 = np.random.normal(2, 1, 100)
y1 = np.random.normal(1, 0.1, 100)
x2 = np.random.normal(-1, 0.4, 20)
y2 = np.random.normal(1.8, 0.4, 20)
'''
plt.scatter(x1, y1)
plt.scatter(x2, y2, cmap="r")
plt.legend(["class 1", "class 2"])
plt.show()


class Adaline:
    def __init__(self, n_iterations=100, random_state=42, learning_rate=0.01):
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.cost=[]
    def fit(self, X, Y):
        rgen = np.random.RandomState(self.random_state)
        self.weights = rgen.normal(loc=0.0, scale=0.01, size=1 + 2)
        for _ in range (self.n_iterations):
            errs=[]
            for x,y in zip(X,Y):
                output = x @ self.weights[1:] + self.weights[0]
                errors = y - output
                errs.append(errors)
                self.weights[1:] = self.weights[1:] + self.learning_rate * np.array(x) * errors
                self.weights[0] = self.weights[0] + self.learning_rate * errors
            self.cost.append(0.5*sum(np.array(errs)**2))



X1 = list(zip(x1, y1))
Y1 = [1] * 100
X2 = list(zip(x2, y2))
Y2 = [-1] * 100
df=pd.DataFrame({'X1':X1})
df=df["X1"].append(pd.Series(X2),ignore_index=True)
df=pd.DataFrame(df)
df.loc[:,'label'] = pd.Series(Y1+Y2)
df=df.sample(frac=1)
adaline = Adaline()
adaline.fit(df.iloc[:,0], df["label"])
plt.axline((0,-adaline.weights[0]/adaline.weights[2]),slope=-adaline.weights[1]/adaline.weights[2],color="r")
plt.scatter(x1, y1)
plt.scatter(x2, y2, cmap="r")
plt.show()
plt.plot(adaline.cost)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.show()
print(adaline.weights)
