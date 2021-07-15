import numpy as np
import random
import time

##hyper parameter setting
lr = 1e-3
K = 5000
m = 10000
n = 500

#function definition
def sigmoid(x):
    x=x.astype(np.float128)
    return 1 / (1 + np.exp(-x))

def derivative_sigmoid(x):
    return sigmoid(x) / (1-sigmoid(x))

def loss_func(yhat, y):
    delta = 1e-10
    return - (y * np.log(yhat + delta) + (1 - y) * np.log( 1 - yhat +delta) )

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
        
def layer_2_learning(x1, x2, y, test_x1, test_x2, test_y):
    global lr, K
    # W = np.random.rand(2,2,1)
    # b = np.random.rand(2,1)
    W = [np.random.rand(1,2), np.random.rand(1,1)]
    b = [np.random.rand(1,1), np.random.rand(1,1)]
    X = np.array([x1,x2])
    Y = np.array(y)
    # Z = [0,0]
    # A = [0,0]
    # dW = [0,0]
    # db = [0,0]
    # dZ = [0,0]
    A = [np.zeros((1,m)), np.zeros((1,m))]
    Z = [np.zeros((1,m)), np.zeros((1,m))]
    dA = [np.zeros((1,m)), np.zeros((1,m))]
    dZ = [np.zeros((1,m)), np.zeros((1,m))]
    db = [np.zeros((1,1)), np.zeros((1))]
    dW = [np.zeros((1,2)), np.zeros((1,1))]
    
    test_X = np.array([test_x1, test_x2])
    test_Y = np.array(test_y)
    test_Z = [0,0]
    test_A = [0,0]

    for i in range(K):
        #forward prop
        Z[0] = np.dot(W[0],X) + b[0]
        A[0] = sigmoid(Z[0])
        Z[1] = np.dot(W[1][0].T,A[0]) + b[1]
        A[1] = sigmoid(Z[1])
        train_loss = loss_func(A[1], Y)
        #backwrad prop
        dZ[1] = A[1] - Y
        dW[1] = (1/m) * np.dot(dZ[1] , A[0].T)
        db[1] = (1/m) * np.sum(dZ[1]) 
        dZ[0] = (W[1][0].T * dZ[1]) * derivative_sigmoid(dZ[1])
        dW[0] = (1/m) * np.dot(dZ[0] ,X.T)
        db[0] = (1/m) * np.sum(dZ[0])        
        # #application
        W[1] -= lr * dW[1]
        b[1] -= lr * db[1]
        W[0] -= lr * dW[0]
        b[0] -= lr * db[0]

        if (i+1)%50 == 0 :
            # #calculate testing loss
            # test_Z[0] = np.dot(W[0].T,test_X) + b[0]
            # test_A[0] = sigmoid(test_Z[0])
            # test_Z[1] = np.dot(W[1][0].T,test_A[0]) + b[1]
            # test_A[1] = sigmoid(test_Z[1])
            # test_loss = loss_func(test_A[1],test_Y)
            print("##%dth learning"%(i+1))
            print("Training Loss : %f"%(np.mean(train_loss)))
            # print("Testing Loss : %f"%(np.mean(test_loss)))
            print("w1 : ", (W[0]))
            print("w2 : ",(W[1]))
            print("b : ",(b[0]))
    A = np.around(A[1])
    test_A = np.around(test_A[1])
    return [(A == Y).mean() * 100.0, (test_A == test_Y).mean() * 100.0]
        



def main():
    #generate m train samples, n test samples
    x1_train, x2_train, y_train = making_dataset(m)
    x1_test, x2_test, y_test = making_dataset(n)

    v_start = time.time()
    v_acc = layer_2_learning(x1_train, x2_train, y_train, x1_test, x2_test, y_test)
    v_end = time.time()
    print("layer_2_learning train accuracy = %.2f"%v_acc[0]+ "%")
    print("layer_2_learning test accuracy = %.2f"%v_acc[1]+ "%")
    print("layer_2_learning time : " , (v_end - v_start))



if __name__ == "__main__":
    main()
