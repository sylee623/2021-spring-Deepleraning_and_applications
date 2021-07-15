import numpy as np
import random
import time
import os


print("#####################################START#####################################")
##hyper parameter setting
lr = 1e-3
K = 5000
m = 10000
n = 500

def sigmoid(x):
    x=x.astype(np.float128)
    return 1/(1+np.exp(-x))
    
def derivative_sigmoid(x):
    return sigmoid(x) / (1-sigmoid(x))

def loss_func(yhat, y):
    delta = 1e-10
    return (y * np.log(yhat + delta) + (1 - y) * np.log( 1 - yhat +delta) )

def testing(test_x1, test_x2, test_y, W, b):
    test_X = np.array([test_x1, test_x2])
    test_Y = np.array(test_y)
    test_Z = [np.zeros((3,m)), np.zeros((1,m))]
    test_A = [np.zeros((3,m)), np.zeros((1,m))]

    test_Z[0] = np.dot(W[0], test_X) + b[0]
    test_A[0] = sigmoid(test_Z[0])
    test_Z[1] = np.dot(W[1], test_A[0]) + b[1]
    test_A[1] = sigmoid(test_Z[1])
    test_loss = -(1/n) * np.sum(loss_func(test_A[1], test_Y))

    output = np.around(test_A[1])
    
    return np.sum(output == test_Y), test_loss


def layer_2_learning(x1, x2, y):
    global lr, K
    np.random.seed(0)
    W = [np.random.rand(1,2), np.random.rand(1,1)]
    b = [np.random.rand(1,1), np.random.rand(1,1)]
    X = np.array([x1,x2])
    Y = np.array(y)

    A = [np.zeros((1,m)), np.zeros((1,m))]
    Z = [np.zeros((1,m)), np.zeros((1,m))]
    dA = [np.zeros((1,m)), np.zeros((1,m))]
    dZ = [np.zeros((1,m)), np.zeros((1,m))]
    db = [np.zeros((1,1)), np.zeros((1))]
    dW = [np.zeros((1,2)), np.zeros((1,1))]

    for i in range(K):
        #forward prop
        Z[0] = np.dot(W[0],X) + b[0]
        A[0] = sigmoid(Z[0])
        Z[1] = np.dot(W[1][0].T,A[0]) + b[1]
        A[1] = sigmoid(Z[1])
        train_loss = - (1/m) * np.sum(loss_func(A[1], Y))
        #backwrad prop
        dZ[1] = A[1] - Y
        dW[1] = (1/m) * np.dot(dZ[1] , A[0].T)
        db[1] = (1/m) * np.sum(dZ[1]) 
        dZ[0] = (W[1].T * dZ[1]) * derivative_sigmoid(dZ[1])
        dW[0] = (1/m) * np.dot(dZ[0] ,X.T)
        db[0] = (1/m) * np.sum(dZ[0])        
        # #application
        W[1] -= lr * dW[1]
        b[1] -= lr * db[1]
        W[0] -= lr * dW[0]
        b[0] -= lr * db[0]

        if (i+1)%50 == 0 :
            print("##%dth learning"%(i+1))
            print("w1 : ", W[0])
            print("w2 : ", W[1])
            print("b1 : ", b[0])
            print("b2 : ", b[1])
    output = np.around(A[1])
    return W, b, np.sum(output == Y) , train_loss


def main():
    #generate m train samples, n test samples
    
    path = os.path.dirname(os.path.abspath(__file__))
    
    x1_train = open(path + "/train_x1.txt", 'r')
    x1_train = x1_train.readlines()
    x1_train = list(map(float,x1_train))
    x2_train = open(path + "/train_x2.txt", 'r')
    x2_train = x2_train.readlines()
    x2_train = list(map(float,x2_train))
    y_train = open(path + "/train_y.txt", 'r')
    y_train = y_train.readlines()
    y_train = list(map(float,y_train))

    x1_test = open(path + "/test_x1.txt", 'r')
    x1_test = x1_test.readlines()
    x1_test = list(map(float,x1_test))
    x2_test = open(path + "/test_x2.txt", 'r')
    x2_test = x2_test.readlines()
    x2_test = list(map(float,x2_test))
    y_test = open(path + "/test_y.txt", 'r')
    y_test = y_test.readlines()
    y_test = list(map(float,y_test))


    train_start = time.time()
    W, b, output, train_loss = layer_2_learning(x1_train, x2_train, y_train)
    train_end = time.time()

    test_start = time.time()
    test_output, test_loss = testing(x1_test, x2_test, y_test, W, b)
    test_end = time.time()

    print("training time : " , (train_end - train_start))
    print("testing time : " , (test_end - test_start))

    print("training_loss : ", train_loss)
    print("testing_loss : ", test_loss)

    print("training_accuracy :%f , # of correctly predicted : %f "%(float(output / m) * 100 , output))
    print("testing_accuracy :%f , # of correctly predicted : %f "%(float(test_output / n)  * 100, test_output))

if __name__ == "__main__":
    main()
