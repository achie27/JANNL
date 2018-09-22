import pickle
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report

from nn import NeuralNet

# Training details for this model
with open("cost_iris.pkl", "rb") as c:
	cost = pickle.load(c)

with open("acc_iris.pkl", "rb") as c:
	accuracy = pickle.load(c)

# Plot training cost and accuracy vs epochs
plt.figure(1)
plt.plot(list(range(0, 1000, 10)), cost)
plt.ylabel("Training Error/Cost"), plt.xlabel("Epochs")
plt.figure(2)
plt.plot(list(range(0, 1000, 10)), accuracy)
plt.ylabel("Training Accuracy"), plt.xlabel("Epochs")
plt.show()

op_nodes = 3
iris = load_iris()
X = iris.data
y = np.array([np.eye(op_nodes)[i] for i in iris.target])

# Shuffle
random = np.random.permutation(len(X))
X = X[random]
y = y[random]

# Load the model
nnet = NeuralNet(load = True, model_file = "trained.pkl")
op = nnet.predict(X, return_max = True)

print("-"*30+"\nTEST SET ACCURACY\n"+"-"*30)
print(classification_report(y.argmax(axis = 1), op))