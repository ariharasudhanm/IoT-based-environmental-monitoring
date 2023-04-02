from nnfs import datasets
import numpy as np
from nnfs.datasets import spiral_data
import nnfs
from numpy import core
nnfs.init()
import matplotlib.pyplot as plt

# Importance of transposing
inp = np.array([[1, 2, 3, 2.5], 
                [2, 5, -1, 2], 
                [-1.5, 2.7, 3.3, -0.8]])
w = np.array([[0.2, 0.8, -0.5, 1], 
              [0.5, -0.91, 0.26, -0.5], 
              [-0.26, -0.27, 0.17, 0.87]])
b = [2,3,0.5]

w2 = np.array([[0.1, -0.14, 0.5], 
               [-0.5, 0.12, -0.33],
               [-0.44, 0.73, -0.13]])
b2 = [-1, 2, -0.5]


out1 = np.dot(inp,w.T) + b 

out2 = np.dot(out1, w2.T) + b2

# Introducing non linear data
X, y = spiral_data(samples = 100, classes = 3)
print(X.shape, y.shape)
# plt.scatter(X[:,0], X[:,1], c= y, cmap = 'brg')
# plt.show()
# print(X.shape)

# Defining dense layers
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):            # n_inputs are inputs, n_output are output neurons
        self.weights =  0.01*np.random.randn(n_neurons, n_inputs)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs                            # to remember for backpropagation
        self.output = np.dot(inputs, self.weights.T) + self.biases
        
    def backward(self, dvalues):
        self.dweights = np.dot(dvalues.T, self.inputs)
        self.dbiases = np.sum(dvalues, axis = 0, keepdims = True)
        self.inputs = np.dot(dvalues, self.weights)

class activation_Relu:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0,inputs)
    def backward(self, dvalues):
        self.dinputs  = dvalues.copy()   #np.copy(dvalues)
        self.dinputs[self.inputs <= 0] = 0


class activation_softmax:
    def forward(self, inputs):
        self.inputs = inputs
        exp_values = np.exp(inputs-np.max(inputs, axis = 1, keepdims = True))
        self.output = exp_values / np.sum(exp_values, axis =1 , keepdims =True)      # Probabilties 
    
    def backward(self,dvalues):
        self.dinputs = np.empty_like(dvalues)
        
        for index, (single_output, single_dvalues) in enumerate (zip(self.output, dvalues)):
            single_output  = single_output.reshape(-1,1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)


class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss


class Loss_categoricalcrossentropy(Loss):
    def forward(self, y_pred, y_true): # suppose if we have predicted output and ground truth labels
        y_pred_clipped = np.clip(y_pred, np.exp(-7), 1-np.exp(-7)) # Excluding the values less than e-7 and higher that 1=e-7
        
        # slicing the y_pred with the help of y_true as a indices variable
        if len(y_true.shape) == 1:         #if the outputs are not one hot encoded directly mentioning about classes it belongs to 
            true_confidences = y_pred_clipped[range(len(y_pred)), y_true]
 
        elif len(y_true.shape) == 2:       # if the outputs are one hot encoded
            true_confidences = np.sum(y_pred_clipped*y_true, axis = 1)
            
        negative_log_likelihoods = -np.log(true_confidences)    # true confidences are values of 
        return negative_log_likelihoods

    def backward(self, dvalues, y_true):
        labels = len(dvalues[0])
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        self.dinputs = -y_true/dvalues
        self.dinputs = self.dinputs/len(dvalues)


class activation_softmax_loss_categoricalentropy():
    def __init__(self):
        self.activation = activation_softmax()
        self.loss = Loss_categoricalcrossentropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis = 1)

        self.dinputs = dvalues.copy()

        self.dinputs[range(len(dvalues)), y_true] -=1 

        self.dinputs = self.dinputs /len(dvalues)


class optimizer_SGD():
    def __init__(self, learning_rate = 1.0):
        self.learning_rate = learning_rate

    def update_params(self, layer):
        layer.weights += -self.learning_rate*layer.dweights
        layer.biases += -self.learning_rate*layer.dbiases


# Creating a object class of Layer_Dense()
layer1 = Layer_Dense(2, 64)

# create a object of class Relu
activation1 = activation_Relu()

# second layer 
layer2 = Layer_Dense(64,3)

# activation function for second layer
activation2 = activation_softmax()
#loss_activation = activation_softmax_loss_categoricalentropy()

# creating object class for categorical loss cross entropy
loss_function = Loss_categoricalcrossentropy()

# creating optimizer class
#optimizer = optimizer_SGD
 
# for epoch in range(25):
        
# There is already created dataset in the above cells
layer1.forward(X)

# pass through activation function
activation1.forward(layer1.output)

# pass through second layer
layer2.forward(activation1.output)

# pass through second activation layer
activation2.forward(layer2.output)

# computing loss for the forward pass 
loss = loss_function.calculate(activation2.output, y)
#loss = loss_activation.forward(layer2.output,y)


# Accuracy calculation
predictions = np.argmax(activation2.output, axis = 1)
if len(y.shape) == 2:
    y = np.argmax(y, axis = 1)
accuracy = np.mean(predictions == y)
print(loss)

    # if not epoch %100:
    #     print(f'epoch:{epoch}'+
    #           f'acc:{accuracy:.3f}'+
    #           f'loss:{loss:.3f}')



softmax_outputs = np.array([[0.7,0.1,0.2],
                             [0.1,0.5,0.4]
                             [0.02,0.9,0.08]])

class_targets = np.array([0,1,1])

softmax_loss = activation_softmax_loss_categoricalentropy()
softmax_loss.backward(softmax_outputs, class_targets)
dvalues1 = softmax_loss.dinputs

activation = activation_softmax()
activation.output = softmax_outputs
loss = Loss_categoricalcrossentropy()
loss.backward(softmax_outputs, class_targets)
activation.backward(loss.dinputs)
dvalues2.activation.dinputs


