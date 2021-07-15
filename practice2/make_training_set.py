import random
import os

lr = 1e-3
K = 10000
m = 10000
n = 500

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

def main():
    #generate m train samples, n test samples
    x1_train, x2_train, y_train = making_dataset(m)
    x1_test, x2_test, y_test = making_dataset(n)

    path = os.path.dirname(os.path.abspath(__file__))
    
    train_x1 = open(path + "/train_x1.txt", "w")
    for x in x1_train :train_x1.write(str(x) + "\n")
    train_x2 = open(path + "/train_x2.txt", "w")
    for x in x2_train :train_x2.write(str(x) + "\n")
    train_y = open(path + "/train_y.txt", "w")
    for x in y_train :train_y.write(str(x) + "\n")

    test_x1 = open(path + "/test_x1.txt", "w")
    for x in x1_test :test_x1.write(str(x) + "\n")
    test_x2 = open(path + "/test_x2.txt", "w")
    for x in x2_test :test_x2.write(str(x) + "\n")
    test_y = open(path + "/test_y.txt", "w")
    for x in y_test :test_y.write(str(x) + "\n")

if __name__ == "__main__":
    main()
