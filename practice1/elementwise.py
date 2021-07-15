    # for ep in range(K):
    #     J  = []
    #     A = np.array([])
    #     dw1, dw2, db = [], [], []
    #     for i in range(0,len(X)):
    #         #forward prop
    #         z=0
    #         for k in range(2):
    #             z += W[k]*X[i][k]
    #         z += b
    #         A = np.append(A, sigmoid(z))
    #         J += [loss_func(A[i],y[i])]
    #         #backprop
    #         dz = A[i] - y[i]
    #         dw1 += [X[i][0] * dz]
    #         dw2 += [X[i][1] * dz]
    #         db += [dz]
    #     #application
    #     W[0] -= lr*(sum(dw1)/len(dw1))
    #     W[1] -= lr*(sum(dw2)/len(dw2))
    #     b -= lr*(sum(db)/len(db))
    #     #print
    #     if (ep+1)%10 == 0 :
    #         #calculate testing loss
    #         test_loss = []
    #         test_A = np.array([])
    #         for j in range(len(test_X)):
    #             test_z = 0
    #             for k in range(2):
    #                 test_z += W[k]*test_X[j][k]
    #             test_z += b
    #             test_A = np.append(test_A, sigmoid(test_z))
    #             test_loss += [loss_func(test_A[j], test_y[j])]
    #         print("##%dth learning"%(ep+1))
    #         print("Training Loss : ", (sum(J)/len(J)))
    #         print("Testing Loss : ", sum(test_loss)/len(test_loss))
    #         print("w1 : ", (W[0]))
    #         print("w2 : ", (W[1]))
    #         print("b : ", (b))
    # A = np.around(A)
    # Y = np.array(y)
    # test_A = np.around(test_A)
    # test_Y = np.array(test_y)
    # return [(A == Y).mean() * 100.0 , (test_A == test_Y).mean() * 100.0]