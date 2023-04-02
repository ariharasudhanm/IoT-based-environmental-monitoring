import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_gaussian_quantiles
from sklearn.model_selection import learning_curve

np.random.seed(42)

# import nnfs


X,y = make_gaussian_quantiles(cov=0.7,
                                 n_samples=5000, n_features=2,
                                 n_classes=2, random_state=42)
Y = y.reshape(-1,1) # reshaping to adapt our code

print('X is :', X[:5], 'y is:', Y[:5])
plt.scatter(X[:, 0], X[:, 1], c=Y, s=40, cmap=plt.cm.Spectral)
plt.show()

# X = np.array([[ 0.05, -1.1],
#                 [ 1.2, -0.1],
#                 [-0, 0.4 ],
#                 [-0.5, -0.2 ],
#                 [ 1.3,  0.6],
#                 [ 0.5,  1.2 ],
#                 [ 0.4, -0.1],
#                 [ 0.2, -1.6],
#                 [-0.1, -0.1],
#                 [-0.4,  0.09],
#                 [-0.9,  0.3],
#                 [-0.8,  0.2],
#                 [-1.4,  -0.4],
#                 [-0.7, -1.1],
#                 [-0.3, -0.3]])

# Y = np.array([[1],[1],[0],[0], [1], [1], [0], [1], [0], [0],[1], [0], [1], [1], [0]])

print('X is :',X[:5], 'y is:',Y[:5])
# plt.scatter(X[:, 0], X[:, 1],c=Y, s=40, cmap=plt.cm.Spectral)
# plt.show()



def init_params():
    w1 = np.random.normal(scale = 0.5, size = (2,6)) # Changes made for higher dimensional dataset
    w2 = np.random.normal(scale = 0.5, size = (6,4))
    w3 = np.random.normal(scale = 0.5, size = (4,1))
    b1 = np.random.normal(scale = 0.5, size = (1,6))
    b2 = np.random.normal(scale = 0.5, size = (1,4))
    
    return w1, w2, w3, b1, b2

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
    outputs = (sigmoid(inputs))*((1-sigmoid(inputs)))
    return outputs


def calculate(predicted, y):
    predicted = np.max(predicted, axis = 1)
    # print('self.pred is:', self.predicted.shape)
    y = np.max(y, axis = 1)

    # print('predicted is :', self.predicted)
    # print('y is:', self.y)
    loss = np.mean(-y*np.log(predicted) - (1- y)*np.log(1-predicted))
    print('CE loss is :', loss)
    # Accuracy calculation
    predicted[predicted >= 0.5] = 1 
    predicted[predicted < 0.5] = 0
    acc = np.mean(predicted == y)

    return acc, predicted[:6], Y[:6]
    # print('accuracy is:', acc)


def delta_error(predicted, y):
    # predicted = np.max(predicted, axis = 1)
    # y = np.max(y, axis = 1)
    outputs = predicted - y
    return outputs
    # print('delta_error is :', self.outputs)


def forward_propagation(w1, w2, w3, b1, b2, X):
    
    z1 = np.dot(X, w1) + b1
    a1 = tanh(z1)

    z2 = np.dot(a1, w2) + b2
    a2 = tanh(z2)

    z3 = np.dot(a2, w3)
    a3 = sigmoid(z3)


    return z1, a1, b1, z2, a2, b2, z3, a3


def backward_propagation(z1, a1, z2, a2, z3, a3, w1, w2, w3, Y):
    d_error = delta_error(a3, Y)
    del_error = d_error*(d_sigmoid(z3))

    dw3 = np.dot(a2.T, del_error)
    dinp3 = np.dot(del_error, w3.T)*(d_tanh(z2))
    db2 = np.sum(del_error, axis = 1, keepdims= True)

    dw2 = np.dot(a1.T, dinp3)
    dinp2 = np.dot(dinp3, w2.T)*(d_tanh(z1))
    db1 = np.sum(dinp3, axis = 1, keepdims=True)

    dw1 = np.dot(X.T, dinp2) 

    return dw1, dw2, dw3, db1, db2



def update_params(w1, w2, w3, dw1, dw2, dw3, b1, db1, b2, db2, learning_rate):
    w1 = w1- learning_rate*dw1
    w2 = w2- learning_rate*dw2
    w3 = w3- learning_rate*dw3
    b1 = b1- learning_rate*db1
    b2 = b2- learning_rate*db2

    return w1, w2, w3, b1, b2


                 
def gradient_descent(X, Y, iterations, learning_rate):
    w1, w2,w3, b1, b2 = init_params()

    for i in range(iterations):
        z1, a1, b1, z2, a2, b2, z3, a3 = forward_propagation(w1, w2, w3, b1, b2, X)
        dw1, dw2, dw3, db1, db2 = backward_propagation(z1, a1, z2, a2, z3, a3, w1, w2, w3, Y)
        w1, w2, w3, b1, b2 = update_params(w1, w2, w3, dw1, dw2, dw3, b1, db1, b2, db2, learning_rate)
        print('acc, ','predicted, ', 'actual, ', calculate(a2, Y))
        if i % 1000 == 0:
            print('iterations:', i)
            
    return w1, w2, w3, b1, b2




tw1, tw2,tw3, tb1, tb2  = gradient_descent(X, Y, 2000, 0.009)

tz1, ta1,tb1, tz2, ta2, tb2, tz3, ta3 = forward_propagation(tw1, tw2,tw3, tb1, tb2,X)

plt.scatter(X[:, 0], X[:, 1],c=ta3, s=40, cmap=plt.cm.Spectral)
plt.show()
