import sys
import pickle
import numpy as np

class NeuralNet:
	"""
	Builds the architecture of a neural network 

	Public methods :
		train
		predict
		dump
		evaluate

	Private methods :
		_feed_forward
		_backpropagate

	Static mehods:
		MSE
		cross_entropy
		softmax
		sigmoid
		sigmoid_prime
		tanh
		tanh_prime

	"""

	def __init__(self, layer_nodes = [], activation = "sigmoid", \
					output_activation = "sigmoid", loss = "mse", \
						load=False, model_file = None, name = "nn"):
		"""

		Depending on the value of load, it either builds the
		architecture or loads a prebuilt model

		Arguments - 
			layer_nodes -> integer list/tuple 
				Each index represents a layer and the corresponding
				element represents the nodes in that layer

			activation -> String
				The activation function to be used for hidden layers

			output_activation -> String
				The activation function to be used for output layer

			loss -> String
				The loss function to be used for error calculation

			load -> Boolean
				True if the architecture is to be loaded

			model_file -> String
				The address of the dumped model file

			name -> String
				Yep, its name.

		eg -
			nnet = NeuralNet([3, 10, 6, 2])
			
			This will build a neural network with 4 layers with the 
			input layer having 3 nodes, output layer having 2 nodes, 
			and the two hidden layers having 6 & 2 nodes respectively.

			nnet = NeuralNet(load = True, model_file = "iris.pkl")
			
			This will load the architecture saved in iris.pkl.

		"""

		if load:

			if model_file == None :
				print("Specify the model file")
				sys.exit(1)

			try :
				with open(model_file, "rb") as model_pkl :
					model = pickle.load(model_pkl)

					self.loss = model["loss"]
					self.weights = model["weights"]
					self.biases = model["biases"]
					self.hidden_layer_activation = model["activation"]
					self.output_activation = model["op_activation"]
					self.layer_nodes = model["layer_nodes"]
					self.total_layers = len(self.layer_nodes)
					self.name = model_file[ : model_file.find('.pkl')]
					self.dropout = model['dropout']
					self.dropout_layers = model['dropout_layers']

					print("Model loaded")

			except FileNotFoundError:
				print("Incorrect model file address")
				sys.exit(1)

			except TypeError:
				print("This is not a model file")
				sys.exit(1)

			except KeyError:
				print("This is not a model file")
				sys.exit(1)

		else :

			# Error checking
			if not all(isinstance(node, int) for node in layer_nodes) :
				print("Number of layer nodes need to be integers")
				sys.exit(1)

			if activation not in ("sigmoid", "tanh"):
				print(activation +" is not supported")
				sys.exit(1)

			if output_activation not in ("sigmoid", "softmax", "none"):
				print(output_activation +" is not supported")
				sys.exit(1)

			if loss not in ("mse", "crossentropy"):
				print(loss +" is not supported")
				sys.exit(1)

			# Initialise stuff
			self.loss = loss
			self.name = name
			self.dropout = 0.0
			self.dropout_layers = []
			self.layer_nodes = layer_nodes
			self.total_layers = len(layer_nodes)
			self.hidden_layer_activation = activation
			self.output_activation = output_activation
			self.weights, self.biases = [np.zeros(1)], [np.zeros(1)]
			
			# Create the desired shape of the network
			for i in range(1, self.total_layers) :
				a, b = layer_nodes[i-1 : i+1]		
				self.weights.append(np.random.randn(a, b))
				self.biases.append(np.random.randn(1, b))	

			print("Initialised model")
			for x in self.weights:
				print(x.shape)


	def train(self, train_x, target_y, epochs = 100, \
				alpha = 1e-3, reg_para = 0.05, batch_size = 30, \
	 				print_details = True, epochs_bw_details = 10, \
	 					dropout = 0.0, dropout_layers = []):

		"""
		Trains the network on the given input and output data

		Arguments -
			train_x -> Numpy array
				The features of input data. Must have the same
				number of columns as the number of input nodes
			
			target_y -> Numpy array
				The true outputs pertaining to each feature vector.
				Needs to have same number of columns as the number
				of output nodes

			epochs -> Integer
				Total number of times the input data 
				should be feeded forward.

			alpha -> Float / Integer
				The learning rate of the network

			reg_para -> Float / Integer
				The regularisation parameter, aka lambda, for
				weights and biases

			batch_size -> Integer 
				Number of feature vectors that should be 
				feeded forward, and backpropagated, in one go 

			print_details -> Boolean
				True if loss and accuracy should be printed

			epochs_bw_details -> Integer
				Loss and accuracy will be calculated 
				every epochs_bw_details epochs

			dropout -> Float
				The probability that a node will be dropped out 
				from each dropout layer

			dropout_layers -> List / Tuple
				The layers that should have dropout. 
				Input layer is 1

		Returns -> Tuple
			tuple[0] -> Training error every epoch_bw_details epochs
			tuple[1] -> Accuracy every epoch_bw_details epochs

		"""

		# Error checking
		if not len(train_x) == len(target_y):
			print("Different number of rows in train_x and target_y")
			sys.exit(1)

		if dropout > 1.0 or dropout < 0.0:
			print("Dropout needs to be between 0 and 1")
			sys.exit(1)

		error = []	# Stores training error
		accuracy = [] # Stores accuracy values
		train_data_size = len(train_x)

		self.dropout = dropout
		self.dropout_layers = dropout_layers

		for epoch in range(epochs):
			
			# Shuffle the training data
			randomize = np.random.permutation(train_data_size)
			train_x = train_x[randomize]
			target_y = target_y[randomize]			

			# Gradient descent
			for batch in range(0, train_data_size, batch_size):
				z, a, dropout_masks = self._feed_forward(
					train_x[batch : batch + batch_size], 
					training = True
				)
				
				self._backpropagate(
					z, a, target_y[batch : batch + batch_size],
					alpha, reg_para, dropout_masks
				)

			# Find error
			if epoch % epochs_bw_details == 0:
				z, a, _masks = self._feed_forward(train_x)
				
				err = 0
				if self.loss == "mse":
					err = self.MSE(a[-1], target_y)
				elif self.loss == "crossentropy":
					err = self.cross_entropy(a[-1], target_y)

				error.append(err)

				acc = self.evaluate(a[-1], target_y)
				accuracy.append(acc)

				if print_details :
					print("Epoch #{0}".format(epoch))
					print('Loss = {0}'.format(err))
					print('Accuracy = {0}\n'.format(acc))

		return (error, accuracy)


	def predict(self, x, return_max = False):
		"""
		Predicts the outputs for x by passing it through the network

		Arguments -
			x -> Numpy array
				Feature vectors whose value needs to be predicted

			return_max -> Boolean
				Returns the maximum index of each prediction if True 

		Returns -> Numpy array or List based on return_max
			Actual prediction or the maximum index of each prediction

		"""

		op = self._feed_forward(x)[1][-1]

		if return_max :
			return op.argmax(axis = 1)

		return op


	def evaluate(self, y_pred, y_target):
		"""
		Evaluates the accuracy of the model

		Arguments -
			y_pred -> Numpy array
				Predictions by the model

			y_target -> Numpy array
				Actual output

		Returns -> Float
			The accuracy of the model

		"""
		
		if not len(y_pred) == len(y_target) :
			print("Different number of rows in y_pred and y_target")
			sys.exit(1)

		y_pred = y_pred.argmax(axis=1) 
		y_target = y_target.argmax(axis=1)
		
		return np.sum(y_pred == y_target) / len(y_pred)


	def dump(self):
		"""
		Dumps the weights, biases, and activations as a pickle

		"""

		model = {
			"activation" : self.hidden_layer_activation,
			"op_activation" : self.output_activation,
			"weights" : self.weights,
			"biases" : self.biases,
			"layer_nodes" : self.layer_nodes,
			"loss" : self.loss,
			"dropout" : self.dropout,
			"dropout_layers" : self.dropout_layers
		}

		with open(self.name + ".pkl", "wb") as model_pkl:
			pickle.dump(model, model_pkl)

		print("Saved model in "+self.name+".pkl.")


	def _feed_forward(self, batch_x, training=False):
		"""
		Feeds forward the data batch in the neural network
		and returns the activation and z of all the layers

		Arguments - 
			batch_x -> Numpy array
				Input data

			training -> Boolean
				True if this function was called as 
				part of training 

		Returns -> tuple
			tuple[0] -> Zs of all the layers -> Numpy array
			tuple[1] -> Activations of all layers -> Numpy array
			tuple[2] -> Dropout masks -> Dict

		"""

		z = [0]*self.total_layers
		activations = [0]*self.total_layers
		masks = {}

		# feed the input data in the first layer
		activations[0] = batch_x

		# Feed forward
		for i in range(1, self.total_layers):
			z[i] = np.dot(activations[i-1], self.weights[i]) 
			z[i] = z[i] + self.biases[i]
			
			# Apply the activation function
			if i == self.total_layers - 1 :
				if self.output_activation == "sigmoid" :
					activations[i] = self.sigmoid(z[i])
				elif self.output_activation == "softmax":
					activations[i] = self.softmax(z[i])
				elif self.output_activation == "none":
					activations[i] = z[i]
			else :
				if self.hidden_layer_activation == "sigmoid" :
					activations[i] = self.sigmoid(z[i])
				elif self.hidden_layer_activation == "tanh":
					activations[i] = self.tanh(z[i])

			# Do dropout
			if self.dropout > 0.0 and i + 1 in self.dropout_layers:
				if training :
					masks[i] = np.random.binomial(
						1, 1.0 - self.dropout, z[i].shape
					)

					activations[i] *= masks[i]

				else :
					
					# Keep probability
					activations[i] *= (1-self.dropout)


		return (z, activations, masks)


	def _backpropagate(self, z, activs, batch_y, \
						alpha, reg_para, dropout_masks ):
		
		"""
		Backpropagates and updates the weights and biases of the
		neural network

		Arguments -
			z -> Numpy array
				Zs found from the forward feed

			activs -> Numpy array
				Activations of layers from the forward feed

			batch_y -> Numpy array
				Target output for the data that was forward feeded

			alpha -> Float
				Learning rate

			reg_para -> Float
				Regularisation parameter

			dropout_masks -> Dict
				Dropout masks for each dropout layer

		"""

		# Going in reverse
		for rev_layer in range(1, self.total_layers):

			# Calculate delta depending on the layer
			if rev_layer == 1 :
				delta = activs[-1] - batch_y
			else:				
				delta = np.dot(delta, self.weights[-rev_layer+1].T)
				if self.hidden_layer_activation == "sigmoid":
					delta = delta * self.sigmoid_prime(z[-rev_layer])
				elif self.hidden_layer_activation == "tanh":
					delta = delta * self.tanh_prime(z[-rev_layer])

				pos = self.total_layers-rev_layer+1 
				if self.dropout > 0.0 and pos in self.dropout_layers:
					delta = delta * dropout_masks[pos-1]
			
			# Find the partial derivative and regularise it
			dw = np.dot(activs[-rev_layer-1].T, delta)
			dw = dw + reg_para * self.weights[-rev_layer]
					
			# Update weights and biases based on the learning rate
			self.weights[-rev_layer] -= alpha * dw
			self.biases[-rev_layer] -= alpha * np.mean(delta, 0)


	@staticmethod
	def MSE(y_pred, y_target):
		"""
		Mean Squared Error

		"""

		return np.mean((y_target - y_pred)**2)


	@staticmethod
	def cross_entropy(y_pred, y_target):
		"""
		Cross Entropy Error (assuming y_pred has probabilities)

		"""

		size = len(y_pred)
		y_target = y_target.argmax(axis=1)
		return np.sum(-np.log(y_pred[range(size), y_target]))/size


	@staticmethod
	def softmax(prob):
		"""
		Softmax for output layer activation 

		"""

		prob = np.exp(prob- np.max(prob))
		return prob / np.sum(prob)


	@staticmethod
	def tanh_prime(z):
		"""
		Derivative of tanh

		"""

		return 1 - np.power(NeuralNet.tanh(z), 2)


	@staticmethod
	def tanh(z):
		"""
		TanH activation for hidden layers

		"""

		return np.tanh(z)


	@staticmethod
	def sigmoid_prime(z):
		"""
		Derivative of Sigmoid function

		"""

		return NeuralNet.sigmoid(z)*(1-NeuralNet.sigmoid(z))


	@staticmethod
	def sigmoid(z):
		"""
		Sigmoid activation function 

		"""

		return 1/(1 + np.exp(-z))
