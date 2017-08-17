"""network3.py
this code can generate the new data
~~~~~~~~~~~~~~
it looks this one will work, so watch this one first.
A Theano-based program for training and running simple neural
networks.

Supports several layer types (fully connected, convolutional, max
pooling, softmax), and activation functions (sigmoid, tanh, and
rectified linear units, with more easily added).

When run on a CPU, this program is much faster than network.py and
network2.py.  However, unlike network.py and network2.py it can also
be run on a GPU, which makes it faster still.

Because the code is based on Theano, the code is different in many
ways from network.py and network2.py.  However, where possible I have
tried to maintain consistency with the earlier programs.  In
particular, the API is similar to network2.py.  Note that I have
focused on making the code simple, easily readable, and easily
modifiable.  It is not optimized, and omits many desirable features.

This program incorporates ideas from the Theano documentation on
convolutional neural nets (notably,
http://deeplearning.net/tutorial/lenet.html ), from Misha Denil's
implementation of dropout (https://github.com/mdenil/dropout ), and
from Chris Olah (http://colah.github.io ).

Written for Theano 0.6 and 0.7, needs some changes for more recent
versions of Theano.

"""

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

#### Constants
# GPU = False
# if GPU:
#     print "Trying to run under a GPU.  If this is not desired, then modify "+\
#         "network3.py\nto set the GPU flag to False."
#     try: theano.config.device = 'gpu'
#     except: pass # it's already set
#     theano.config.floatX = 'float32'
# else:
#     print "Running with a CPU.  If this is not desired, then the modify "+\
#         "network3.py to set\nthe GPU flag to True."

#### Load the MNIST data
def load_data_shared(filename="test.csv"):
# the problem is I have more [] in every row
    f=open(filename, 'r')  # 73257 for train,26032 for test,but we don't know why we have two more.

    data = f.read()

    f.close()
    rows = data.split('\n')
    n=10000
    dx=10000
    lth =26032+2#531131+2 #73257+2
    bbb = np.zeros((lth - 2, 3072),np.float32)
    ccc= np.zeros((lth - 2, 32*32),np.float32)
    #labley = [0 for i in range(lth-2)]
    split_row = rows[0].split(",")#3073

    for row in range(1, lth - 1):
        split_row = rows[row].split(",")


        bbb[row - 1] = split_row#[1:]
        # if int(split_row[0]) ==10:
        #     #labley[row - 1] = 0
        # else:
            #labley[row - 1]=int(split_row[0])

    #print labley[0:100]
    #b = b.reshape(32, 32, 3, -1).transpose(3, 0, 1, 2) for picture
    #(73257, 32, 32, 3) this is for plot
    #b=b.transpose(0,3,1,2)
    bbb = bbb.reshape(32, 32, 3, -1).transpose(3, 2,0,1)
    #bbb=bbb.transpose(3, 0,1,2)

    newbbb=bbb[:,0]*0.33+bbb[:,1]*0.33+bbb[:,2]*0.34
    newbbb=newbbb.reshape(26032, 1024)
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
    small=np.zeros((8,1024))
    newbbb=np.vstack((newbbb,small))
    f = open('rebuilttestgray.txt', 'wb')
    pickle.dump(newbbb, f)
    f.close()
    #np.savetxt("rebuilttraingray.txt", newbbb);


    #np.savetxt("extrala.txt", labley);
    print("ok")
    #np.savetxt("vi.txt", validation_inputs);
    #np.savetxt("vo.txt", validation_results);
    lu=newbbb[0]*255
    lu=lu.reshape(32,32)
    plt.imshow(lu,cmap=plt.cm.gray)
    plt.show()

    #curr=training_inputs[1].reshape(32,32)
    #curr=curr*255;
    #plt.imshow(curr.astype('uint8'))
    #plt.show()
    #print training_inputs[1].shape

    def shared(data):
        """Place the data into shared variables.  This allows Theano to copy
        the data to the GPU, if one is available.

        """
        shared_x = theano.shared(
            np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
        shared_y = theano.shared(
            np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
        return shared_x, T.cast(shared_y, "int32")
    return [shared(training_data), shared(validation_data), shared(test_data)]

#### Main class used to construct and train networks



load_data_shared()