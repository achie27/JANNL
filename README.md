
# Installation
Following are the dependencies -
```
1. Numpy
2. Scikit-Learn - For testing, cross-validation and metric calculation
3. Matplotlib - Visualising Cost and Accuracy over time
```
Install the python3 dependencies by running -
```
pip3 install -r requirements
```

# Working

There are three python files -
1. pretrained_model.py
2. nn.py
3. iris_model.py

**pretrained_model.py** loads the best performing model I made, plots its Cost vs Epochs and Accuracy vs Epochs graphs, and makes prediction on the whole Iris dataset. Below is the config I used -
```
Architecture -
	layer_nodes = [4, 15, 20, 10, 3]
	activation = sigmoid
	output_activation = sigmoid
	loss = crossentropy
```
```
Training -
	epochs = 1000
	alpha = 1e-3
	regularisation parameter = 0.05
	batch size = 30
	dropout = 0.0
```

I used the following specs for sklearn's train/test split -
```
train_test_split
	test_size = 0.33
	random_state = 42
```

This model achieved a training accuracy of ~0.97, test accuracy of 1.0, and a mean cross validation accuracy of ~0.99. It gave slightly better training accuracies till 2000 epochs but turned stagnant after that.

**nn.py** is the Neural Network library I was tasked to make. A list is passed during object creation which builds the architecture of the network. A *pre-trained model* can also be loaded. The class has four methods - *train*, *predict*, *evaluate*, and *dump* which do as they read. One can chose among *softmax* and *sigmoid* for output activation, *tanh* and *sigmoid* for hidden layer activation, and *MSE* and *Cross Entropy* for loss calculation. *L1 Regularisation* of weights and biases, and *dropout* to reduce overfitting are also options. The file itself is well documented, so I will be leaving out a detailed explanation.

**iris_model.py** shows NeuralNet's API usage. I have also included helper functions from scikit-learn - train_test_split, classification_report, and KFold - which help in selecting the best model.
