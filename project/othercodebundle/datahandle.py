import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
def load_test_shared(filename="test.csv"):
    f=open(filename, 'r')  
    data = f.read()
    f.close()
    rows = data.split('\n')
    lth =26032+2#531131+2 #73257+2
    bbb = np.zeros((lth - 2, 3072),np.float32)
    #ccc= 
    labley = np.zeros((lth - 2, 10),np.float32)
    split_row = rows[0].split(",")#3073
    for row in range(1, lth - 1):
        split_row = rows[row].split(",")
        bbb[row - 1] = split_row
    bbb = bbb.reshape(32, 32, 3, -1).transpose(3, 2,0,1)
    newbbb=bbb[:,0]*0.3+bbb[:,1]*0.59+bbb[:,2]*0.11
    newbbb=newbbb.reshape(26032, 1024)
    newbbb=newbbb/255
    small=np.zeros((8,1024))
    newbbb=np.vstack((newbbb,small))
    f = open('testdateforsub.txt', 'wb')
    pickle.dump(newbbb, f)
    f.close()
    print("testdone")
    return None
def load_train_shared(filename="train.csv"):
    f=open(filename, 'r')  
    data = f.read()
    f.close()
    rows = data.split('\n')
    lth =73257+2#531131+2 #73257+2
    bbb = np.zeros((lth - 2, 3072),np.float32)
    labley = np.zeros((lth - 2, 10),np.float32)
    split_row = rows[0].split(",")#3073
    for row in range(1, lth - 1):
        split_row = rows[row].split(",")
        bbb[row - 1] = split_row[1:]
        if int(split_row[0]) ==10:
            labley[row - 1,0] = 0
        else:
            labley[row - 1,int(split_row[0])]=1
    bbb = bbb.reshape(32, 32, 3, -1).transpose(3, 2,0,1)
    newbbb=bbb[:,0]*0.3+bbb[:,1]*0.59+bbb[:,2]*0.11
    newbbb=newbbb.reshape(73257, 1024)
    newbbb=newbbb/255
    f = open('traindateforsub.txt', 'wb')
    pickle.dump(newbbb, f)
    f.close()
    f = open('traindateforsubla.txt', 'wb')
    pickle.dump(labley, f)
    f.close()
    print("traindone")
    return None

def load_extra_shared(filename="extra.csv"):
    f=open(filename, 'r')  
    data = f.read()
    f.close()
    rows = data.split('\n')
    lth =531131+2
    bbb = np.zeros((lth - 2, 3072),np.float32)
    labley = np.zeros((lth - 2, 10),np.float32)
    split_row = rows[0].split(",")#3073
    for row in range(1, lth - 1):
        split_row = rows[row].split(",")
        bbb[row - 1] = split_row[1:]
        if int(split_row[0]) ==10:
            labley[row - 1,0] = 0
        else:
            labley[row - 1,int(split_row[0])]=1
    bbb = bbb.reshape(32, 32, 3, -1).transpose(3, 2,0,1)
    newbbb=bbb[:,0]*0.3+bbb[:,1]*0.59+bbb[:,2]*0.11
    newbbb=newbbb.reshape(531131, 1024)
    newbbb=newbbb/255
## below is to segment the lable extradataset and dump them into binary code.
    for i in range(0,11):
        ss=str(i)
        fname="hahah/e"+ss+".txt"
        f=open(fname,'wb')
        pickle.dump(newbbb[50000*i:50000*(i+1)], f)
        f.close()

    
    for i in range(0,11):
        ss=str(i)
        fname="hahah/l"+ss+".txt"
        f=open(fname,'wb')
        pickle.dump(labley[50000*i:50000*(i+1)], f)
        f.close()
    print("extradone")
    return None

def plot100():
    plt.figure(figsize=(8,8),dpi=300)
    for k in range(10):
        for j in range(10):
            curr = bbb[10*k+j].reshape(32, 32)
            curr = curr * 255;
            axes=plt.subplot(10, 10, 10 * (k) + j + 1)
            plt.imshow(curr,cmap=cm.gray)
            axes.set_xticks([])
            axes.set_yticks([]) 
    #plt.show();
    plt.savefig("final1.png")
    return None








load_test_shared()
load_train_shared()
os.mkdir("hahah")
load_extra_shared()




