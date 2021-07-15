import numpy as np
import random
import time
import os

##hyper parameter setting
lr = 1e-3
K = 5000
m = 10000
n = 500

#function definition
def sigmoid(x):
    x=x.astype(np.float128)
    return 1 / (1 + np.exp(-x))

def loss_func(yhat, y):
    delta = 1e-10
    return (y * np.log(yhat + delta) + (1 - y) * np.log( 1 - yhat +delta) )
        
def testing(x1, x2, y ,W, b):
    X = np.array([x1,x2])
    Y = np.array(y)

    Z = np.dot(W.T,X) + b
    A = sigmoid(Z)
    test_loss = (-1/m) * (np.sum(loss_func(A,Y)))
    
    output = np.around(A)
    
    return np.sum(output == Y), test_loss


def vectorized_learning(x1, x2, y):
    global lr, K
    np.random.seed(0)
    W = np.random.rand(2,1)
    b = np.random.rand(1,1)
    X = np.array([x1,x2])
    Y = np.array(y)
    dw1, dw2, db = 0, 0, 0

    for i in range(K):
        #forward prop
        Z = np.dot(W.T,X) + b
        A = sigmoid(Z)
        train_loss = (-1/m) * (np.sum(loss_func(A,Y)))
        #forward prop
        dZ = A - Y
        dW = (1/m) * (X @ dZ.T)
        db = (1/m) * np.sum(dZ)
        # #application
        W -= lr * dW
        b -= lr * db
        if (i+1)%50 == 0 :
            #calculate testing loss
            print("##%dth learning"%(i+1))
            print("w1 : %f"%(W[0]))
            print("w2 : %f"%(W[1]))
            print("b : %f"%(b))
    output = np.around(A)
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
    W, b, output, train_loss = vectorized_learning(x1_train, x2_train, y_train)
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
