#without extra is 94.28 with extra is 93.1
#it indeicate it is only 90 but actually it is 95.
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
#from tensorflow.examples.tutorials.mnist import input_data
import pickle
import tensorflow as tf
import time

a1=time.time()
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#i can'g remember the parameters. Maybe the lr is 2e-4 many be 1e-4 maybe 1.5e-4 epoch many between 10 to 12 or 20 with extradata get 94%
FLAGS = None
cover=0; #hide the left and right, 1 or 0
flip=0; #change black to whtie , 1 and 0
repeat=1; #repeat=2 is not good, 1 or 2
extra=1;
recover=0;
ep=1 # maybe this is too large
epex=1 #10 i enough but maybe 15 is better
hidden=512
print ("loading bbb")
#f = open('newgray.txt','rb')
f = open('train0321.txt','rb')
bbb = pickle.load(f)#,encoding="latin1"
f.close()

# 0 to1
print ("loading lable")

f = open('newlabley2.txt','rb')#newlabley2
labley = pickle.load(f)
f.close()  # 0 to1
print(labley[1])



print(len(bbb))
 

#if os.path.isfile('trainhistory.txt'=1):
#    f = open('valihistory.txt','r')#newlabley2
#    valiresult = pickle.load(f)
#    f.close() 
#    f = open('trainhistory.txt','r')
#    trainsult = pickle.load(f)
#    f.close()
#else:
valiresult=[]
trainsult=[]




f = open('test0321.txt','rb')
testdatax = pickle.load(f)
testdatax=testdatax*255
f.close()  # 0 to1
print(np.min(testdatax),np.max(testdatax))
compare = np.loadtxt('compare.txt')
print( testdatax.shape)


if flip==1:
    for i in range(len(bbb)):
        a = np.average(bbb[i])
        ll = bbb[i] >= a
        if sum(ll) > 512:
            bbb[i] = 1 - bbb[i]
    for i in range(len(testdatax)):
        a = np.average(testdatax[i])
        ll = testdatax[i] >= a
        if sum(ll) > 512:
            testdatax[i] = 1 - testdatax[i]

bbb=bbb[0:73200]*255
labley=labley[0:73200]


#bbb=np.vstack((bbb,-bbb))
#labley=np.vstack((labley,labley))

for i in range(len(bbb)):
    #print (np.average(bbb[i]))
    #print (np.max(bbb[i]))
    bbb[i]=bbb[i]-np.average(bbb[i])
    bbb[i]=bbb[i]/np.std(bbb[i])
    #bbb[i] = bbb[i]/np.std(bbb[i])
# bbb=bbb-np.mean(bbb)
# bbb=bbb/np.std(bbb)

for i in range(len(testdatax)):
    #print (np.average(bbb[i]))
    #print (np.max(bbb[i]))
    testdatax[i]=testdatax[i]-np.average(testdatax[i])
    if np.std(testdatax[i])>1e-3:
        testdatax[i]=testdatax[i]/np.std(testdatax[i])
# testdatax=testdatax-np.mean(testdatax)
# testdatax=testdatax/np.std(testdatax)
print(np.max(bbb))
print(np.max(testdatax))
x = tf.placeholder(tf.float32, [None, 32*32])
W = tf.Variable(tf.zeros([32*32, 10]))
b = tf.Variable(tf.zeros([10]))
  #y = tf.matmul(x, W) + b
y = tf.nn.softmax(tf.matmul(x, W) + b)
  # Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 10])



def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):

  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
def relu(x):
    return tf.maximum(0.1*x,x)

def frac_pool(x):
    return tf.nn.fractional_max_pool(x, [1.0,1.414,1.414,1.0], pseudo_random=True,overlapping=True)

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1,32,32,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_conv11=tf.nn.lrn(h_conv1)
# h_pool1 = max_pool_2x2(h_conv1)
#h_pool1 = frac_pool(h_conv1)

W_conv12 = weight_variable([5, 5, 32, 32])
b_conv12 = bias_variable([32])
h_conv12 = tf.nn.relu(conv2d(h_conv11, W_conv12) + b_conv12)
h_conv121=tf.nn.lrn(h_conv12)
#h_pool2 = max_pool_2x2(h_conv2)

W_conv13 = weight_variable([3, 3, 32, 32])
b_conv13 = bias_variable([32])
h_conv13 = tf.nn.relu(conv2d(h_conv121, W_conv13) + b_conv13)
h_conv131=tf.nn.lrn(h_conv13)
h_pool13 = frac_pool(h_conv131+h_conv11)[0]







W_conv2 = weight_variable([3, 3, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool13, W_conv2) + b_conv2)
h_conv21=tf.nn.lrn(h_conv2)
# h_pool2 = max_pool_2x2(h_conv2)
W_conv22 = weight_variable([3, 3, 64, 64])
b_conv22 = bias_variable([64])
h_conv22 = tf.nn.relu(conv2d(h_conv21, W_conv22) + b_conv22)
h_conv221=tf.nn.lrn(h_conv22)

W_conv23 = weight_variable([3, 3, 64, 64])
b_conv23 = bias_variable([64])
h_conv23 = tf.nn.relu(conv2d(h_conv221, W_conv23) + b_conv23)
h_conv231=tf.nn.lrn(h_conv23)
h_pool23 = frac_pool(h_conv231+h_conv21)[0]







W_conv3 = weight_variable([3, 3, 64, 128])
b_conv3 = bias_variable([128])
h_conv3 = tf.nn.relu(conv2d(h_pool23, W_conv3) + b_conv3)
h_conv31=tf.nn.lrn(h_conv3)
#h_pool3 = max_pool_2x2(h_conv31)

W_conv32 = weight_variable([3, 3, 128, 128])
b_conv32 = bias_variable([128])
h_conv32 = tf.nn.relu(conv2d(h_conv31, W_conv32) + b_conv32)
h_conv321=tf.nn.lrn(h_conv32)



W_conv4 = weight_variable([3, 3, 128, 128])   #89 as 92
b_conv4 = bias_variable([128])
h_conv4 = tf.nn.relu(conv2d(h_conv321, W_conv4) + b_conv4)
h_conv41=tf.nn.lrn(h_conv4)
h_poo321 = frac_pool(h_conv41+h_conv31)[0]#h_pool4 = max_pool_2x2(h_conv41)

W_conv42 = weight_variable([3, 3, 128, 256])   #89 as 92
b_conv42 = bias_variable([256])
h_conv42 = tf.nn.relu(conv2d(h_poo321, W_conv42) + b_conv42)
h_conv412=tf.nn.lrn(h_conv42)

W_conv5 = weight_variable([3, 3, 256, 256]) #91 as 94  #95 97 98
b_conv5 = bias_variable([256])
h_conv5 = tf.nn.relu(conv2d(h_conv412, W_conv5) + b_conv5)
h_conv51=tf.nn.lrn(h_conv5)
h_poo51 = frac_pool(h_conv51)[0]






W_fc1 = weight_variable([49 * 256, 2048])
b_fc1 = bias_variable([2048])
h_pool2_flat = tf.reshape(h_poo51, [-1, 49*256])
h_fc1 = relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)




W_fc2 = weight_variable([2048, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

batchsize=40
batchsizetest=100


global_step = tf.Variable(0, trainable=False)
boundaries = [int(73000*2/batchsize*3+500000/batchsize*0),int(73000*2/batchsize*3+500000/batchsize*1),int(73000*2/batchsize*3+500000/batchsize*3),int(73000*2/batchsize*3+500000/batchsize*6)]
values = [0.2e-4,0.18e-4,0.17e-4,0.16e-4,0.15e-4]
#values = [1.5e-4,1e-4,0.8e-4,0.6e-4,0.4e-4] #i changed it form 0.4e-4 to 1.5e-4
learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)



cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
#train_step = tf.train.AdamOptimizer(0.15e-4).minimize(cross_entropy,global_step=global_step)#1e-4 changed to 1e-3
train_step = tf.train.AdamOptimizer(0.4e-4).minimize(cross_entropy)#1e-4 changed to 1e-3



#1.5e-4 and epoch=7 can get 95 percent
y_p = tf.argmax(y_conv,1)
correct_prediction = tf.equal(y_p, tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()


saver.restore(sess, "codebackup/remodel678.ckpt")







startn=73000
dn=100

pred=np.zeros((26040,10))
for j in range(int(ep)):
  print(j)
  for i in range(int(73000/batchsize)):
  #batch = mnist.train.next_batch(50)

    if i== 0:
      train_accuracy = accuracy.eval(feed_dict={
          x:bbb[batchsizetest * i:batchsizetest * i + batchsizetest], y_: labley[batchsizetest * i:batchsizetest * i + batchsizetest], keep_prob: 1})

      print("step %d, training accuracy %g"%(i, train_accuracy))
      #trainsult.append(train_accuracy)

    train_step.run(feed_dict={x: bbb[batchsize * i:batchsize * i + batchsize], y_: labley[batchsize * i:batchsize * i + batchsize], keep_prob: 0.7})#0.5 changed to 1
    # print(np.array(tf.argmax(y_,1)))
    #print(len(tf.argmax(y_,1)))
#  for i in range(int(73200/batchsize),int((73200+73000)/batchsize)):
#  #batch = mnist.train.next_batch(50)
#
#    if i== 0:
#      train_accuracy = accuracy.eval(feed_dict={
#          x:bbb[batchsizetest * i:batchsizetest * i + batchsizetest], y_: labley[batchsizetest * i:batchsizetest * i + batchsizetest], keep_prob: 1})
#
#      print("step %d, training accuracy %g"%(i, train_accuracy))
#      #trainsult.append(train_accuracy)
#
#    train_step.run(feed_dict={x: bbb[batchsize * i:batchsize * i + batchsize], y_: labley[batchsize * i:batchsize * i + batchsize], keep_prob: 0.7})#0.5 changed to 1
#    # print(np.array(tf.argmax(y_,1)))
    #print(len(tf.argmax(y_,1)))
  a=[]
  for ll in range(2):
    a.append(accuracy.eval(feed_dict={
    x: bbb[startn+ll*100:startn+dn+100*ll], y_: labley[startn+100*ll:startn+dn+100*ll], keep_prob: 1}));
  print(np.mean(a))
  if np.mean(a)>=0.965:
      print("good")
      for i in range(int(26040 / 40)):
          pred[40*i:40*i+40] = pred[40*i:40*i+40] +sess.run(y_conv, feed_dict={x: testdatax[40 * i:40 * i + 40], keep_prob: 1})
      
  



  valiresult.append(a)
#     ("test accuracy %g"%accuracy.eval(feed_dict={
#    x: testdatax[0:100], y_: compare[0:100], keep_prob: 1.0}))
# print
if extra==1:
    for l in range(epex):

        print(epex)
        for k in range(10):

            ss=str(k)
            print(k)
            fname="extra/e"+ss+".txt"
            f = open(fname, 'rb')
            exdata = pickle.load(f)
            exdata=exdata*255
            f.close()

            fname = "extra/l" + ss + ".txt"
            f = open(fname, 'rb')
            la = pickle.load(f)
            f.close()
            exdata=np.vstack((exdata,-exdata))
            la=np.vstack((la,la))
            for i in range(len(exdata)):
                # print (np.average(bbb[i]))
                # print (np.max(bbb[i]))
                exdata[i] = exdata[i] - np.average(exdata[i])
                exdata[i] = exdata[i] / np.std(exdata[i])
            for i in range(int(100000 / batchsize)):


                train_step.run(
                    feed_dict={x: exdata[batchsize * i:batchsize * i + batchsize], y_: la[batchsize * i:batchsize * i + batchsize],
                        keep_prob: 0.7})  # 0.5 changed to 1
            a = []
            for ll in range(2):
                a.append(accuracy.eval(feed_dict={
                    x: bbb[startn + ll * 100:startn + dn + 100 * ll], y_: labley[startn + 100 * ll:startn + dn + 100 * ll],
                    keep_prob: 1}));
            print(np.mean(a))
            if np.mean(a)>=0.965:
                print("good")
                for i in range(int(26040 / 40)):
                    pred[40*i:40*i+40] = pred[40*i:40*i+40] +sess.run(y_conv, feed_dict={x: testdatax[40 * i:40 * i + 40], keep_prob: 1})
      


        for i in range(int(73000/ batchsize)):
            # batch = mnist.train.next_batch(50)



            train_step.run(feed_dict={x: bbb[batchsize * i:batchsize * i + batchsize],
                                      y_: labley[batchsize * i:batchsize * i + batchsize],
                                      keep_prob: 0.7})  # 0.5 changed to 1
        for i in range(int(73200/batchsize),int((73200+73000)/batchsize)):
  #batch = mnist.train.next_batch(50)

            if i== 0:
                train_accuracy = accuracy.eval(feed_dict={
                        x:bbb[batchsizetest * i:batchsizetest * i + batchsizetest], y_: labley[batchsizetest * i:batchsizetest * i + batchsizetest], keep_prob: 1})

        
      #trainsult.append(train_accuracy)

            train_step.run(feed_dict={x: bbb[batchsize * i:batchsize * i + batchsize], y_: labley[batchsize * i:batchsize * i + batchsize], keep_prob: 0.7})#0.5 changed to 1
    # print(np.array(tf.argmax(y_,1)))
    #print(len(tf.argmax(y_,1)))

        a = []
        for ll in range(2):
            a.append(accuracy.eval(feed_dict={
                x: bbb[startn + ll * 100:startn + dn + 100 * ll], y_: labley[startn + 100 * ll:startn + dn + 100 * ll],
                keep_prob: 1}));
        print(np.mean(a))
        
        if np.mean(a)>=0.97:
            print("good")
            for i in range(int(26040 / 40)):
                pred[40*i:40*i+40] = pred[40*i:40*i+40] +sess.run(y_conv, feed_dict={x: testdatax[40 * i:40 * i + 40], keep_prob: 1})
      
        
#        pred=np.zeros((26040,10))
#        for i in range(int(26040 / 40)):
#            pred[40*i:40*i+40] = pred[40*i:40*i+40] +sess.run(y_conv, feed_dict={x: testdatax[40 * i:40 * i + 40], keep_prob: 1})
#            # pred = np.asarray(pred)
            
        result = np.argmax(pred,axis=1)
        print(result)
        result = result.reshape(1, 26040)
        result[result == 0] = 10
        twe = np.zeros((2, 26032))
        twe[0] = [i for i in range(0, 26032)]
        twe[1] = result[0, 0:26032]
        twe = twe.T.astype(int)
        sstitle=str(l)
        fname = "predit0415" + ss + ".csv"
        csvfile = open(fname, 'w')
        writer = csv.writer(csvfile)
        writer.writerow(['ImageId', 'Label'])
        writer.writerows(twe)
        csvfile.close()
        
saver.save(sess, "codebackup/remodel678.ckpt")
print('saved')
#f = open('trainhistory.txt','w')#newlabley2
##pickle.dump(trainsult,f)
#f.close() 
#f = open('valihistory.txt','w')#newlabley2
#pickle.dump(valiresult,f)
#f.close() 


#plt.plot(trainsult)
#plt.plot(valiresult)
#plt.show()
result=[]
pred=np.zeros((26040,10))
for i in range(int(26040 / 40)):
    pred[40*i:40*i+40] = pred[40*i:40*i+40] +sess.run(y_conv, feed_dict={x: testdatax[40 * i:40 * i + 40], keep_prob: 1})+sess.run(y_conv, feed_dict={x: -testdatax[40 * i:40 * i + 40], keep_prob: 1})
            # pred = np.asarray(pred)
print("ok")          
result = np.argmax(pred,axis=1)
print(result)
result = result.reshape(1, 26040)
result[result == 0] = 10



print(result.shape)
happy=0;
plotnumber=[0 for i in range(20)]
ll=0;
for i in range(100):
    if result[0,i]==compare[i]:
        happy=happy+1;
    else:

        plotnumber[ll]=i
        #print(result[0,i])
        ll=ll+1
print("happ", happy)
for k in range(2):
    for j in range(10):
        curr = testdatax[plotnumber[10*k+j]].reshape(32, 32)
        curr = curr * 256;
        plt.subplot(2, 10, 10 * (k) + j + 1)
        plt.imshow(curr,cmap=cm.gray)

        plt.title("{0}".format(result[0,plotnumber[10*k+j]]) )

plt.show();

twe = np.zeros((2, 26032))
twe[0] = [i for i in range(0, 26032)]
twe[1] = result[0,0:26032]
twe=twe.T.astype(int)


csvfile = open('predit0301.csv', 'w')
writer = csv.writer(csvfile)
writer.writerow(['ImageId','Label'])
writer.writerows(twe)
csvfile.close()

# -*- coding: utf-8 -*-
h_conv11,W_conv1
print(W_conv2.eval().shape)
plit=W_conv2.eval().transpose(3,2,0,1) #(5,5,1,32)

print(plit.shape)
plt.figure(figsize=(8,4),dpi=300)
for k in range(4):
    for j in range(8):
        curr = plit[8*k+j][0].reshape(3,3)
        #curr = curr * 256;
        #plt.subplot(8, 8, 8 * (k) + j + 1)
        #plt.imshow(curr,cmap=cm.gray)
        axes=plt.subplot(4, 8, 8 * (k) + j + 1)
        plt.imshow(curr,cmap=cm.gray)
        axes.set_xticks([])
        axes.set_yticks([]) 
        #plt.title("{0}".format(result[0,plotnumber[10*k+j]]) )

plt.show();


h_conv1
i=0
pred=sess.run(h_conv1, feed_dict={x: testdatax[40 * i:40 * i + 40], keep_prob: 1})
curr=pred[0]
curra = pred.reshape(40,32,32,32).transpose(0,3,1,2)
print(h_conv1)
testa=testdatax[0]
testa=testa.reshape(32,32)
plt.imshow(testa,cmap=cm.gray)
plt.figure(figsize=(8,4),dpi=300)
for k in range(4):
    for j in range(8):
        currab = curra[0][8*k+j].reshape(32,32)
        #curr = curr * 256;
        #plt.subplot(8, 8, 8 * (k) + j + 1)
        #plt.imshow(curr,cmap=cm.gray)
        axes=plt.subplot(4, 8, 8 * (k) + j + 1)
        plt.imshow(currab,cmap=cm.gray)
        axes.set_xticks([])
        axes.set_yticks([]) 
        
        
        
        h_pool23
        
        
pred=sess.run(y_p, feed_dict={x: bbb[73000:73200], keep_prob: 1})
print(h_poo51.shape)
curra = pred.reshape(40,7,7,256).transpose(0,3,1,2)

plt.imshow(testa,cmap=cm.gray)
plt.figure(figsize=(8,8),dpi=300)
for k in range(8):
    for j in range(8):
        currab = curra[0][8*k+j].reshape(7,7)
        #curr = curr * 256;
        #plt.subplot(8, 8, 8 * (k) + j + 1)
        #plt.imshow(curr,cmap=cm.gray)
        axes=plt.subplot(8, 8, 8 * (k) + j + 1)
        plt.imshow(currab,cmap=cm.gray)
        axes.set_xticks([])
        axes.set_yticks([]) 
        
print ("Precision", sk.metrics.precision_score(y_true, y_pred))

cvlable=np.zeros((200,1))
for i in range(200):
    cvlable[i]=np.argmax(labley[73000+i])
    pred=pred.reshape(200,1)

aaa=[]
bbbb=[]
for i in range(200):
    aaa.append(int(cvlable[i]))
    bbbb.append(int(pred[i]))
cccc=tf.contrib.metrics.confusion_matrix(aaa, bbbb).eval()
plt.imshow(cccc)
plt.cm.gist_rainbow