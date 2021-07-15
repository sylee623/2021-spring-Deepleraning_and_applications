import numpy as np
import os
import time
import random

m = 10000 # num of train sample
n = 500  # num of evaluation sample
iterations = 10000
log_step = 50
alpha = 1e-3  # Hyper Parameter
flag_print = True

W1_initial = np.random.randn(3,2)
b1_initial = np.random.randn(3,1)
W2_initial = np.random.randn(1,3)
b2_initial = np.random.randn(1,1)


"""
Functions for logistic regression for vectorized version
"""
def cross_entropy_loss(y_hat, y):
    #pdb.set_trace()
    a1 = (y * np.log(y_hat))
    a2 = (1 - y) * np.log(1 - y_hat + 1e-10)
    return a1 + a2

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def model(x, W, b):
    return np.dot(W, x) + b


def train_n_test(x_train, y_train, x_test, y_test):

    # Initialize Fucntion Parameters
    W1 = W1_initial
    b1 = b1_initial
    W2 = W2_initial
    b2 = b2_initial

    acc_train = 0
    acc_test = 0
    cost_train = 0
    cost_test = 0

    start_time = time.time()
    if flag_print:
        print("\n\nInitial Function Parameters: ", W1, b1, W2, b2)
        print("\n######### Training #########")
    for iteration in range(iterations):
        # Foward Propagation
        Z1 = model(x_train, W1, b1)
        A1 = sigmoid(Z1)
        Z2 = model(A1, W2, b2)
        A2 = sigmoid(Z2)
        cost = np.sum((-cross_entropy_loss(A2, y_train))) / m

        # Backward Propagation
        dZ2 = A2 - y_train
        dW2 = np.dot(dZ2, A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m

        dZ1 = np.dot(W2.T, dZ2) * (A1 * (1 - A1))
        dW1 = np.dot(dZ1, x_train.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m

        # Calculate Accuracy
        y_hat = A2
        y_hat[y_hat > 0.5] = 1
        y_hat[y_hat <= 0.5] = 0
        acc = np.sum(y_hat == y_train)

        if (iteration + 1) % log_step == 0 and flag_print:
            print("%d iteration => Cost: %f, Training Accuracy: %f%%" % (iteration + 1, cost, acc / m * 100.0))
        acc_train = (acc / m * 100.0)
        cost_train = cost

        # Parameters Update
        W1 = W1 - alpha * dW1
        b1 = b1 - alpha * db1
        W2 = W2 - alpha * dW2
        b2 = b2 - alpha * db2
        
    end_time = time.time()
    train_time = (end_time - start_time) / iterations

    start_time = time.time()
    if flag_print:
        print("\n######### Inference #########")
    Z1 = model(x_test, W1, b1)
    A1 = sigmoid(Z1)
    Z2 = model(A1, W2, b2)
    A2 = sigmoid(Z2)
    cost = np.sum((-cross_entropy_loss(A2, y_test))) / n

    y_hat = A2
    y_hat[y_hat > 0.5] = 1
    y_hat[y_hat <= 0.5] = 0
    acc = np.sum(y_hat == y_test)

    if flag_print:
        print("Cost: %f, Test Accuracy: %f%%" % (cost, acc / n * 100.0))
    acc_test = acc / n * 100.0
    cost_test = cost
    
    end_time = time.time()
    test_time = end_time - start_time

    return train_time, test_time, acc_train, acc_test, cost_train, cost_test

def making_dataset(num):
    x1 ,x2 ,y =[], [], []
    for i in range(num):
        x1.append(random.uniform(-10,10))
        x2.append(random.uniform(-10,10))
        if(x1[-1] + x2[-1] > 0):
            y.append(1)
        else:
            y.append(0)
    return x1, x2, y

if __name__ == "__main__":

    x1_train, x2_train, y_train = making_dataset(m)
    x1_test, x2_test, y_test = making_dataset(n)

    x_train = np.array([x1_train, x2_train])
    y_train = np.array(y_train)

    x_test = np.array([x1_test, x2_test])
    y_test = np.array(y_test)
    
    T_train, T_test, acc_train, acc_test, cost_train, cost_test = train_n_test(x_train, y_train, x_test, y_test)
    print("\n\n")
    print("######## HYPER-PARAMETERS ########")
    print("num of train sample (m) : %d" % (m))
    print("num of test sample (n) : %d" % (n))
    print("num of iterations (k) : %d" % (iterations))
    print("\n######## TASK 3 RESULT ########")
    print("Training Time : %.6f" % (T_train))
    print("Training Accuracy : %f" % (acc_train))
    print("Training Cost : %f" % (cost_train))
    print("Test Time : %.6f" % (T_test))
    print("Test Accuracy : %f" % (acc_test))
    print("Test Cost : %f" % (cost_test))