
#### Libraries
# Standard library
import pickle
import gzip

# Third-party libraries
import numpy as np
# import theano
# import theano.tensor as T
# from theano.tensor.nnet import conv
# from theano.tensor.nnet import softmax
# from theano.tensor import shared_randomstreams
# from theano.tensor.signal import downsample

# Activation functions for neurons
def linear(z): return z
# def ReLU(z): return T.maximum(0.0, z)
# from theano.tensor.nnet import sigmoid
# from theano.tensor import tanh
import matplotlib.pyplot as plt


def load_data_shared(filename="test.csv"):
# the problem is I have more [] in every row
    f=open(filename, 'r')  # 73257 for train,26032 for test,but we don't know why we have two more.

    data = f.read()

    f.close()
    rows = data.split('\n')
    n=10000
    dx=10000
    lth =26032+2#26032+2#531131+2 #73257+2
    bbb = np.zeros((lth - 2, 3072),np.float32)
    ccc= np.zeros((lth - 2, 32*32),np.float32)
    #labley = [0 for i in range(lth-2)]
    split_row = rows[0].split(",")#3073
    xxx=np.zeros((lth - 2, 3072),np.float32)
    for row in range(1, lth - 1):
        split_row = rows[row].split(",")


        xxx[row - 1] = split_row
        bbb[row-1]=xxx[row-1]#[1:]
        # if int(split_row[0]) ==10:
        #     #labley[row - 1] = 0
        # else:
            #labley[row - 1]=int(split_row[0])

    #print labley[0:100]
    bbb=np.array(bbb)
    #b = b.reshape(32, 32, 3, -1).transpose(3, 0, 1, 2) for picture
    #(73257, 32, 32, 3) this is for plot
    #b=b.transpose(0,3,1,2)
    bbb = bbb.reshape(32, 32, 3, -1).transpose(3, 2,0,1)
    #bbb=bbb.transpose(3, 0,1,2)

    newbbb=bbb[:,0]*0.299+bbb[:,1]*0.587+bbb[:,2]*0.114
    newbbb=newbbb.reshape(lth-2, 1024)
    # print bbb.shape
    # print bbb[0].shape
    # print bbb[0][0].shape

    # for i in range(10):
    #     for j in range(10):
    #         plt.subplot(10, 10, 10 * (i) + j + 1)
    #         plt.imshow(newbbb[10 * (i) + j + 1].astype('uint8'))
    # plt.show()
    #b=b.reshape(lth-2,3072)
    newbbb=newbbb/255
    small=np.zeros((8,1024))
    newbbb=np.vstack((newbbb,small))
    #print "begin traing data"



    # training_results=[0 for i in range(n)]
    # validation_results=[0 for i in range(dx)]
    # test_results=[0 for i in range(dx)]
    # validation_inputs=bbb[n:n+dx,0:32*32]
    # validation_results[0:dx]=labley[n:n+dx]
    # test_inputs=bbb[n+dx:n+2*dx,0:32*32]
    # test_results[0:dx]=labley[n+dx:n+2*dx]
    # training_inputs=bbb[0:n,0:32*32]
    # training_results[0:n]=labley[0:n]
    # validation_results=np.array(validation_results,dtype=np.float32)
    # test_results = np.array(test_results, dtype=np.float32)
    # training_results = np.array(training_results, dtype=np.float32)
    # training_data = (training_inputs, training_results)
    # validation_data=(validation_inputs, validation_results)
    # test_data = (test_inputs, test_results)
    # print "wrint"
    #small=np.zeros((8,1024))
    #newbbb=np.vstack((newbbb,small))
    f = open('test0321.txt', 'wb')
    pickle.dump(newbbb, f)
    f.close()
    #np.savetxt("rebuilttraingray.txt", newbbb);


    #np.savetxt("extrala.txt", labley);
    print("ok")
    #np.savetxt("vi.txt", validation_inputs);
    #np.savetxt("vo.txt", validation_results);
    # lu=newbbb[0]*255
    # lu=lu.reshape(32,32)
    # plt.imshow(lu,cmap=plt.cm.gray)
    # plt.show()


    bbb = bbb.reshape(-1,3,32, 32).transpose(0,2,3,1)
    # xxx=bbb[1]
    # print(np.max(bbb),np.min(bbb))
    # plt.imshow(xxx)
    # plt.show()


    for i in range(10):
        for j in range(10):
            plt.subplot(10, 10, 10 * (i) + j + 1)
            plt.imshow(bbb[10 * (i) + j + 1].astype('uint8'))
    plt.show()
load_data_shared()