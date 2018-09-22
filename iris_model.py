import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, KFold

from nn import NeuralNet

# Load dataset
op_nodes = 3
iris = load_iris()
x = iris.data
y = np.array([np.eye(op_nodes)[i] for i in iris.target])

# Split into training and testing set
# Using 40% of the dataset for testing purposes
x_train, x_test, y_train, y_test = train_test_split(
	x, y, test_size = 0.4, random_state = 27
)

# Config
layer_nodes = [4, 10, 3]	# Add more elements for more layers
model_name = "iris2"
activation = "sigmoid"				# (sigmoid, tanh)
op_activation = "sigmoid"			# (sigmoid, softmax)
loss = "crossentropy"				# (crossentropy, mse)

# Build the architecture
nnet = NeuralNet(
	layer_nodes = layer_nodes, name = model_name, loss = loss,
	activation = activation, output_activation = op_activation 
)

# Training config
epochs = 2000
alpha = 1e-3
reg_para = 0.05
batch_size = 20
epochs_bw_details = 50
dropout_percent = 0.25		# Probability of a node dropping out
d_layers = [2]				# Only these layers will have dropout

# Training
cost, accuracy = nnet.train(
	x_train, y_train, epochs = epochs, alpha = alpha,
	reg_para = reg_para, batch_size = batch_size,
	print_details = True, epochs_bw_details = epochs_bw_details,
	dropout = dropout_percent, dropout_layers = d_layers
)

# Cost and Accuracy plotting
plt.figure(1)
plt.ylabel("Training Error/Cost"), plt.xlabel("Epochs")
plt.plot(list(range(0, epochs, epochs_bw_details)), cost)
plt.figure(2)
plt.plot(list(range(0, epochs, epochs_bw_details)), accuracy)
plt.ylabel("Training Accuracy"), plt.xlabel("Epochs")
plt.show()

op = nnet.predict(x_test, return_max = True)

print("-"*30+"\nTEST SET ACCURACY\n"+"-"*30)
print(classification_report(
	y_test.argmax(axis=1), op 
), '\n')


splits = 5
cumulative_acc = 0

# k-Fold Cross validation
kf = KFold(n_splits = splits, shuffle = True)
for i, (train_index, test_index) in enumerate(kf.split(x)):
	x_train, x_test = x[train_index], x[test_index]
	y_train, y_test = y[train_index], y[test_index]

	_cost, _accuracy = nnet.train(
		x_train, y_train, epochs = epochs, alpha = alpha,
		reg_para = reg_para, batch_size = batch_size,
		print_details = False, epochs_bw_details = epochs_bw_details,
		dropout = dropout_percent, dropout_layers = d_layers
	)

	op = nnet.predict(x_test)

	cumulative_acc += nnet.evaluate(op, y_test)
	print("-"*30+"\nKFold #{0}\n".format(i)+"-"*30)
	print(classification_report(y_test.argmax(axis=1), op.argmax(axis=1)))
	print()

print("Mean cross validation accuracy = {0}".format(cumulative_acc/splits))

s = input("Dump model? (y/n)\n")
if s=='y':
	nnet.dump()