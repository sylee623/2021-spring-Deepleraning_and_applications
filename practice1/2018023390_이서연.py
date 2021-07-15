import numpy as np
import random
import time

##hyper parameter setting
lr = 1e-2
K = 2000
m = 1000
n = 100

#function definition
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

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

def elementwise_learning(x1, x2, y, test_x1, test_x2, test_y):
    global lr, K, m, n
    W = np.random.rand(2,1)
    b = np.random.rand(1)
    Y = np.array(y)
    
    for ep in range(K):
        a = []
        train_loss = 0
        db , dw1, dw2, dz = 0, 0, 0, 0
        for i in range(m):
            z = W[0]*x1[m] + W[1]*x2
            z = z + b
            a += [sigmoid(z)[0]]
            print(a[i])
            train_loss += (loss_func(a[i],y[i]))
            dz = a[i] - y[i]
            dw1 += x1[i] * dz
            dw2 += x2[i] * dz
            db += dz
        train_loss /= m
        dw1 /= m
        dw2 /= m
        db /= m
        W[0] = W[0] - lr * dw1
        W[1] = W[1] - lr * dw2
        b = b - lr * db
        if((ep+1)%10 == 0):
            test_a = []
            test_loss = 0
            for j in range(n):
                z = W[0] * test_x1[j] + W[1] * test_x2[j] + b
                test_a.append(sigmoid(z)[0])
                test_loss += (loss_func(test_a[j],test_y[j]))
            test_loss /= n
            print("##%dth learning"%(ep+1))
            print("Training Loss : %f"%(train_loss))
            print("Testing Loss : %f"%(test_loss))
            print("w1 : %f"%(W[0]))
            print("w2 : %f"%(W[1]))
            print("b : %f"%(b))
    A = np.array(a)
    A = np.around(A)
    test_A = np.array(test_a)
    test_A = np.around(test_A)
    Y = np.array(y)
    test_Y = np.array(test_y)
    return [(A == Y).mean() * 100.0, (test_A == test_Y).mean() * 100.0]

        
def vectorized_learning(x1, x2, y, test_x1, test_x2, test_y):
    global lr, K
    W = np.zeros(2)
    b = 0
    X = np.array([x1,x2])
    Y = np.array(y)
    dw1, dw2, db = 0, 0, 0
    
    test_X = np.array([test_x1, test_x2])
    test_Y = np.array(test_y)

    for i in range(K):
        #forward prop
        Z = np.dot(W.T,X) + b
        A = sigmoid(Z)
        train_loss = (-1/m)*(np.sum((Y*np.log(A)) + ((1-Y)*(np.log(1-A)))))
        #forward prop
        dZ = A - Y
        dW = (1/m) * (X @ dZ.T)
        print(dW)
        break
        db = (1/m) * np.sum(dZ)
        # #application
        W -= lr * dW
        b -= lr * db
        if (i+1)%10 == 0 :
            #calculate testing loss
            test_Z = np.dot(W.T, test_X)+b
            test_A = sigmoid(test_Z)
            test_loss = loss_func(test_A, test_Y)          
            print("##%dth learning"%(i+1))
            print("Training Loss : %f"%(np.mean(train_loss)))
            print("Testing Loss : %f"%(np.mean(test_loss)))
            print("w1 : %f"%(W[0]))
            print("w2 : %f"%(W[1]))
            print("b : %f"%(b))
    A = np.around(A)
    test_A = np.around(test_A)
    return [(A == Y).mean() * 100.0, (test_A == test_Y).mean() * 100.0]
        



def main():
    #generate m train samples, n test samples
    x1_train, x2_train, y_train = making_dataset(m)
    x1_test, x2_test, y_test = making_dataset(n)
    
    # ew_start = time.time()
    # e_acc = elementwise_learning(x1_train, x2_train, y_train, x1_test, x2_test, y_test)
    # ew_end = time.time()

    v_start = time.time()
    v_acc = vectorized_learning(x1_train, x2_train, y_train, x1_test, x2_test, y_test)
    v_end = time.time()
    # print("elementwise train accuracy = %.2f"%e_acc[0]+ "%")
    # print("elementwise test accuracy = %.2f"%e_acc[1]+ "%")
    print("vectorized train accuracy = %.2f"%v_acc[0]+ "%")
    print("vectorized test accuracy = %.2f"%v_acc[1]+ "%")
    # print("elementwise time : " , (ew_end - ew_start))
    print("vectorized time : " , (v_end - v_start))



if __name__ == "__main__":
    main()
