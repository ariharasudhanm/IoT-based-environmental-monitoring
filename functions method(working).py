import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_gaussian_quantiles
from sklearn.model_selection import learning_curve
import sklearn
from sklearn import linear_model
np.random.seed(42)


# Data creation and visualization
# X,y = make_gaussian_quantiles(cov=0.3, n_samples=150, n_features=2,n_classes=2, random_state=42)
# Y = y.reshape(-1,1) # reshaping to adapt our code
# print('X is :',X.shape, 'Y is:',Y.shape)
# plt.scatter(X[:, 0], X[:, 1],c=y, s=40, cmap=plt.cm.Spectral)
# plt.show()

X = np.array([[0,1.0], 
              [1.0,1.0],
              [1.0,0]])

Y = np.array([[1.0],
              [0],
              [1.0]])


def init_params():
    w1 = np.random.randn(2,3) # Changes made for higher dimensional dataset
    w2 = np.random.randn(3,1)
    return w1, w2


def tanh(inputs):
    outputs = np.tanh(inputs)
    return outputs

def d_tanh(dvalues):
    outputs = 1-np.tanh(dvalues)**2
    return outputs

def sigmoid(inputs):
    outputs = 1/(1+np.exp(-inputs))
    return outputs

def d_sigmoid(inputs):
    outputs = sigmoid(inputs)*(1-sigmoid(inputs))
    return outputs


def calculate(predicted, y):
    predicted = np.max(predicted, axis = 1)
    # print('self.pred is:', self.predicted.shape)
    y = np.max(y, axis = 1)

    # print('predicted is :', self.predicted)
    # print('y is:', self.y)
    loss = -np.mean(y*np.log(predicted) + (1-y)*np.log(1-predicted))
    print('CE loss is :',loss)
    # Accuracy calculation
    # predicted[predicted >= 0.5] = 1 
    # predicted[predicted < 0.5] = 0
    acc = np.mean(predicted == y)
    return acc, predicted, Y
    # print('accuracy is:', acc)

def delta_error(predicted, y):
    # self.predicted = np.max(predicted, axis = 1)
    # self.y = np.max(y, axis = 1)
    outputs = predicted - y
    return outputs
    # print('delta_error is :', self.outputs)


def forward_propagation(w1, w2, X):
    
    z1 = np.dot(X,w1)
    a1 = tanh(z1)

    z2 = np.dot(a1, w2)
    a2 = sigmoid(z2)

    return z1, a1, z2, a2


def backward_propagation(z1, a1, z2, a2, w1, w2, Y):
    d_error = delta_error(a2, Y)
    del_error = d_error*d_sigmoid(z2)

    dw2 = np.dot(a1.T, del_error)
    dinp2 = np.dot(del_error, w2.T)*d_tanh(z1)

    dw1 = np.dot(X.T, dinp2) # Changes made for higher dimensional dataset
    return dw1, dw2



def update_params(w1, w2,  dw1, dw2, learning_rate):
    w1 = w1- learning_rate*dw1
    w2 = w2- learning_rate*dw2

    return w1, w2



def gradient_descent(X, Y, iterations, learning_rate):
    w1, w2 = init_params()

    for i in range(iterations):
        z1, a1, z2, a2 = forward_propagation(w1, w2, X)
        dw1, dw2 = backward_propagation(z1, a1, z2, a2,  w1, w2, Y)
        w1, w2 = update_params(w1, w2,  dw1, dw2, learning_rate)
        print('acc :', calculate(a2, Y))
        if i % 200 == 0:
            print('iterations:', i)
            
    return w1, w2


gradient_descent(X, Y, 4000, 0.09)

