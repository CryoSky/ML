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
from tensorflow.examples.tutorials.mnist import input_data
import pickle
import tensorflow as tf
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#i can'g remember the parameters. Maybe the lr is 2e-4 many be 1e-4 maybe 1.5e-4 epoch many between 10 to 12 or 20 with extradata get 94%
FLAGS = None
cover=0; #hide the left and right, 1 or 0
flip=0; #change black to whtie , 1 and 0
repeat=2; #repeat=2 is not good, 1 or 2
extra=1;
ep=3
epex=10
hidden=256
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

vali = bbb[73000:73200]#,encoding="latin1"


# 0 to1


valila = labley[73000:73200]
bbb=bbb[0:73000]
labley=labley[0:73000]
bbb=np.vstack((bbb,-bbb))
labley = np.vstack((labley, labley))




if repeat==2:
    bbb = np.vstack((bbb, -bbb))
    labley = np.vstack((labley, labley))
#print("maxis",labley[0:10])
f = open('test0321.txt','rb')
testdatax = pickle.load(f)

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


if cover==1:
    bbb = bbb.reshape(-1, 32, 32)
    for i in range(len(bbb)):
        summ=np.sum(bbb[i],axis=0)
    #print (summ.shape)
        avg=np.average(bbb[i])
        if np.sum(summ[6:26])<0.8*np.sum(summ):
            bbb[i,:,0:6]=avg
            bbb[i,:,26:]=avg
    bbb = bbb.reshape(-1, 1024)
for i in range(len(bbb)):
    #print (np.average(bbb[i]))
    #print (np.max(bbb[i]))
    bbb[i]=bbb[i]-np.average(bbb[i])
    #bbb[i] = bbb[i]/np.std(bbb[i])
# bbb=bbb-np.mean(bbb)
# bbb=bbb/np.std(bbb)
if cover==1 :
    testdatax = testdatax.reshape(-1, 32, 32)
    for i in range(len(testdatax)):
        summ=np.sum(testdatax[i],axis=0)
        avg=np.average(testdatax[i])
        if np.sum(summ[6:26])<0.8*np.sum(summ):
            testdatax[i,:,0:6]=avg
            testdatax[i,:,26:]=avg

    testdatax=testdatax.reshape(-1,1024)
for i in range(len(testdatax)):
    #print (np.average(bbb[i]))
    #print (np.max(bbb[i]))
    testdatax[i]=testdatax[i]-np.average(testdatax[i])
    #testdatax[i]=testdatax[i]/np.std(testdatax[i])
# testdatax=testdatax-np.mean(testdatax)
# testdatax=testdatax/np.std(testdatax)
print(np.max(bbb))
x = tf.placeholder(tf.float32, [None, 32*32])
W = tf.Variable(tf.zeros([32*32, 10]))
b = tf.Variable(tf.zeros([10]))
  #y = tf.matmul(x, W) + b
y = tf.nn.softmax(tf.matmul(x, W) + b)
  # Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 10])

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.
# cross_entropy = tf.reduce_mean(
#       tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
#
# for i in range(500):
#   batch_xs, batch_ys =bbb[100*i:100*i+100],labley[100*i:100*i+100] #mnist.train.next_batch(100)
#   sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
#
#   # Test trained model
# correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# print(sess.run(accuracy, feed_dict={x: bbb[60000:61000],#mnist.test.images,
#                                       y_: labley[60000:61000]}))


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

def frac_pool(x):
    return tf.nn.fractional_max_pool(x, [1.0,1.414,1.414,1.0], pseudo_random=True,overlapping=True)
def relu(x):
    return tf.maximum(0.1*x,x)


W_conv1 = weight_variable([3, 3, 1, 64])
b_conv1 = bias_variable([64])
x_image = tf.reshape(x, [-1,32,32,1])

h_conv1 = relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
#h_pool1 = frac_pool(h_conv1)
W_conv12 = weight_variable([3, 3, 64, 128])
b_conv12 = bias_variable([128])
h_conv12 = relu(conv2d(h_pool1, W_conv12) + b_conv12)

W_conv2 = weight_variable([3, 3, 128, 128])
b_conv2 = bias_variable([128])
h_conv2 = relu(conv2d(h_conv12, W_conv2) + b_conv2)
#h_pool2 = max_pool_2x2(h_conv2)


W_conv3 = weight_variable([3, 3, 128, 128])
b_conv3 = bias_variable([128])
h_conv3 = relu(conv2d(h_conv2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)

W_conv4 = weight_variable([3, 3, 128, 128])   #89 as 92
b_conv4 = bias_variable([128])
h_conv4 = relu(conv2d(h_pool3, W_conv4) + b_conv4)
h_pool3 = max_pool_2x2(h_conv4)

W_conv5 = weight_variable([3, 3, 128, 256]) #91 as 94  #95 97 98
b_conv5 = bias_variable([256])
h_conv5 = relu(conv2d(h_conv4, W_conv5) + b_conv5)



W_fc1 = weight_variable([16 * 256, hidden])
b_fc1 = bias_variable([hidden])
h_pool2_flat = tf.reshape(h_conv5, [-1, 16*256])
h_fc1 = relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


# W_fc3 = weight_variable([1024, 300])
# b_fc3 = bias_variable([10])
# y_conv = tf.matmul(h_fc1_drop, W_fc3) + b_fc3


W_fc2 = weight_variable([hidden, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


batchsize=40
batchsizetest=40

global_step = tf.Variable(0, trainable=False)
boundaries = [int(73000*2/batchsize*3+100000/batchsize*0),int(73000*2/batchsize*3+100000/batchsize*1),int(73000*2/batchsize*3+100000/batchsize*3),int(73000*2/batchsize*3+100000/batchsize*6)]
values = [1.5e-4,1.3e-4,1e-4,0.8e-4,0.4e-4] #i changed it form 0.4e-4 to 1.5e-4
learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)

#1.5e-4 and epoch=7 can get 95 percent
y_p = tf.argmax(y_conv,1)
correct_prediction = tf.equal(y_p, tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())
startn=70000
dn=40



valiresult=[]
trainsult=[]

for j in range(int(ep)):
  print(j)
  for i in range(int(73000*2/batchsize)):
  #batch = mnist.train.next_batch(50)

    if i== 0:
      train_accuracy = accuracy.eval(feed_dict={
          x:bbb[batchsizetest * i:batchsizetest * i + batchsizetest], y_: labley[batchsizetest * i:batchsizetest * i + batchsizetest], keep_prob: 1})

      print("step %d, training accuracy %g"%(i, train_accuracy))
      trainsult.append(train_accuracy)

    train_step.run(feed_dict={x: bbb[batchsize * i:batchsize * i + batchsize], y_: labley[batchsize * i:batchsize * i + batchsize], keep_prob: 0.5})#0.5 changed to 1
    # print(np.array(tf.argmax(y_,1)))
    #print(len(tf.argmax(y_,1)))
  a = accuracy.eval(feed_dict={
      x: vali[0:400], y_: valila[0:400], keep_prob: 1});
  print(a)
  #valiresult.append(a)
    #print("test accuracy %g"%accuracy.eval(feed_dict={
   #x: testdatax[0:100], y_: compare[0:100], keep_prob: 1.0}))

if extra==1:
    for l in range(epex):
        for k in range(10
                       ):

            ss=str(k)
            print(k)
            fname="extra/e"+ss+".txt"
            f = open(fname, 'rb')
            exdata = pickle.load(f)
            f.close()
            fname = "extra/l" + ss + ".txt"
            f = open(fname, 'rb')
            la = pickle.load(f)
            f.close()
            for i in range(len(exdata)):
                # print (np.average(bbb[i]))
                # print (np.max(bbb[i]))
                exdata[i] = exdata[i] - np.average(exdata[i])
            for i in range(int(50000 / batchsize)):


                train_step.run(
                    feed_dict={x: exdata[batchsize * i:batchsize * i + batchsize], y_: la[batchsize * i:batchsize * i + batchsize],
                        keep_prob: 0.5})  # 0.5 changed to 1

    a = accuracy.eval(feed_dict={
        x: vali[0:400], y_: valila[0:400], keep_prob: 1});
    print(a)

result=[]

#
#plt.plot(trainsult)
#plt.plot(valiresult)
#plt.show()
for i in range(int(26040/40)):
    pred = sess.run(y_p, feed_dict={x: testdatax[40*i:40*i+40], keep_prob: 1})
    #pred = np.asarray(pred)
    result.append(pred)

result=np.array(result)
print(result)
result=result.reshape(1,26040)
result[result==0]=10
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

