import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import input_data
import matplotlib.pyplot as plt
#matplotlib inline

def unpickle(file):
    import pickle
    fo = open(file, 'rb')
    dict = pickle.load(fo)
    fo.close()
    return dict

currentCifar = 1
cifartr = unpickle('train0321.txt')
for i in range(len(cifartr)):
    #print (np.average(bbb[i]))
    #print (np.max(bbb[i]))
    cifartr[i]=cifartr[i]-np.average(cifartr[i])
    cifartr[i]=cifartr[i]/np.std(cifartr[i])

cifarTla = unpickle('newlabley2.txt')
cifarTla=np.argmax(cifarTla,axis=1)
print("down")
print(cifarTla)


total_layers = 25 #Specify how deep we want our network
units_between_stride = int(total_layers / 5)

def resUnit(input_layer,i):
    with tf.variable_scope("res_unit"+str(i)):
        part1 = slim.batch_norm(input_layer,activation_fn=None)
        part2 = tf.nn.relu(part1)
        part3 = slim.conv2d(part2,64,[3,3],activation_fn=None)
        part4 = slim.batch_norm(part3,activation_fn=None)
        part5 = tf.nn.relu(part4)
        part6 = slim.conv2d(part5,64,[3,3],activation_fn=None)
        output = input_layer + part6
        return output

tf.reset_default_graph()

input_layer = tf.placeholder(shape=[None,32,32,1],dtype=tf.float32,name='input')
label_layer = tf.placeholder(shape=[None],dtype=tf.int32)
label_oh = slim.layers.one_hot_encoding(label_layer,10)

layer1 = slim.conv2d(input_layer,64,[3,3],normalizer_fn=slim.batch_norm,scope='conv_'+str(0))
for i in range(5):
    for j in range(units_between_stride):
        layer1 = resUnit(layer1,j + (i*units_between_stride))
    layer1 = slim.conv2d(layer1,64,[3,3],stride=[2,2],normalizer_fn=slim.batch_norm,scope='conv_s_'+str(i))
    
top = slim.conv2d(layer1,10,[3,3],normalizer_fn=slim.batch_norm,activation_fn=None,scope='conv_top')

output = slim.layers.softmax(slim.layers.flatten(top))

loss = tf.reduce_mean(-tf.reduce_sum(label_oh * tf.log(output) + 1e-10, axis=[1]))
trainer = tf.train.AdamOptimizer(learning_rate=0.001)
update = trainer.minimize(loss)



init = tf.global_variables_initializer()


batch_size = 64
currentCifar = 1
total_steps = 20000
l = []
a = []
aT = []
with tf.Session() as sess:
    sess.run(init)
    saver = tf.train.Saver()
    #saver.restore(sess, "codebackup/resnigh1.ckpt")
    i = 0
    draw = range(10000)
    while i < total_steps:
        if i % (10000/batch_size) != 0:
            batch_index = np.random.choice(draw,size=batch_size,replace=False)
        else:
            draw = range(10000)
#             if currentCifar == 5:
#                 currentCifar = 1
#                 print ("Switched CIFAR set to " + str(currentCifar))
#             else:
#                 currentCifar = currentCifar + 1
#                 print ("Switched CIFAR set to " + str(currentCifar))
#             cifar = unpickle('./cifar10/data_batch_'+str(currentCifar))
            batch_index = np.random.choice(draw,size=batch_size,replace=False)
        x = cifartr[batch_index]
        x = np.reshape(x,[batch_size,32,32,1],order='F')
        
        y = np.reshape(np.array(cifarTla)[batch_index],[batch_size,1])
        _,lossA,yP,LO = sess.run([update,loss,output,label_oh],feed_dict={input_layer:x,label_layer:np.hstack(y)})
        accuracy = np.sum(np.equal(np.hstack(y),np.argmax(yP,1)))/float(len(y))
        l.append(lossA)
        a.append(accuracy)
        if i % 10 == 0: print ("Step: " + str(i) + " Loss: " + str(lossA) + " Accuracy: " + str(accuracy))
#         if i % 100 == 0: 
#             point = np.random.randint(0,10000-500)
#             xT = cifarT['data'][point:point+500]
#             xT = np.reshape(xT,[500,32,32,3],order='F')
#             xT = (xT/256.0)
#             xT = (xT - np.mean(xT,axis=0)) / np.std(xT,axis=0)
#             yT = np.reshape(np.array(cifarT['labels'])[point:point+500],[500])
#             lossT,yP = sess.run([loss,output],feed_dict={input_layer:xT,label_layer:yT})
#             accuracy = np.sum(np.equal(yT,np.argmax(yP,1)))/float(len(yT))
#             aT.append(accuracy)
#             print( "Test set accuracy: " + str(accuracy))
        i+= 1
        saver.save(sess, "codebackup/resnigh1.ckpt")