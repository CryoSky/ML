from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import gzip
import csv
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import argparse
import sys
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import pickle
import tensorflow as tf
import random
f = open('train0321.txt','rb')
bbb = pickle.load(f)#,encoding="latin1"
f.close()

f = open('newlabley2.txt','rb')#newlabley2
labley = pickle.load(f)
f.close()
l1=[]
l2=[]
l3=[]
l4=[]
l5=[]
l6=[]
l7=[]
l8=[]
l9=[]
l0=[]

for i in range(len(bbb)):
    if labley[i][1]==1:
        l1.append(i)
    elif labley[i][2]==1:
        l2.append(i)
    elif labley[i][3] == 1:
        l3.append(i)
    elif labley[i][4] == 1:
        l4.append(i)
    elif labley[i][5] == 1:
        l5.append(i)
    elif labley[i][6] == 1:
        l6.append(i)
    elif labley[i][7] == 1:
        l7.append(i)
    elif labley[i][8] == 1:
        l8.append(i)
    elif labley[i][9] == 1:
        l9.append(i)
    elif labley[i][0] == 1:
        l0.append(i)
a=set()

a.update(random.sample(l1, 20))
a.update(random.sample(l2, 20))
a.update(random.sample(l3, 20))
a.update(random.sample(l4, 20))
a.update(random.sample(l5, 20))
a.update(random.sample(l6, 20))
a.update(random.sample(l7, 20))
a.update(random.sample(l8, 20))
a.update(random.sample(l9, 20))
a.update(random.sample(l0, 20))
print(len(a))
valibbb=np.zeros((200,1024))
vala=np.zeros((200,10))
trainbbb=np.zeros((73000,1024))
trly=np.zeros((73000,10))
j=0
for i in a:

    valibbb[j]=bbb[i]
    vala[j]=labley[i]
    j = j + 1
j=0
for i in range(len(bbb)):
    if (i not in a )and(j<73000) :
        trainbbb[j]=bbb[i]
        trly[j]=labley[i]
        j=j+1



valibbb=np.array(valibbb)
valibbb=valibbb.reshape(200,1024)
trainbbb=np.array(trainbbb)[0:73000]
trainbbb=trainbbb.reshape(-1,1024)
vala=np.array(vala)
vala=vala.reshape(200,10)
trly=np.array(trly)[0:73000]
trly=trly.reshape(-1,10)

fname = "extra/e10.txt"
f = open(fname, 'rb')
exdata = pickle.load(f)
f.close()
fname = "extra/l10.txt"
f = open(fname, 'rb')
la = pickle.load(f)
f.close()
valibbb=np.vstack((valibbb,exdata[0:200]))
vala=np.vstack((vala,la[0:200]))

f = open('trainsmall.txt', 'wb')
pickle.dump(trainbbb, f)
f.close()

f = open('trainsmallla.txt', 'wb')
pickle.dump(trly, f)
f.close()

f = open('valismall.txt', 'wb')
pickle.dump(valibbb, f)
f.close()
f = open('valaly.txt', 'wb')
pickle.dump(vala, f)
f.close()


