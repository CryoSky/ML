import numpy as np
import matplotlib.pyplot as plt
import pickle
f = open('newtest.txt','rb')
testdatax = pickle.load(f)
f.close()
bbb=testdatax[0:100]
bbb=bbb.reshape(-1,32,32)
for i in range(10):
    for j in range(10):
        curr = bbb[10 * (i) + j ]*255
        plt.subplot(10,10,10*i+j+1);
        plt.imshow(curr.astype('uint8'))
plt.show()