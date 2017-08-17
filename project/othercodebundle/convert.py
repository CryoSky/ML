import pickle
import numpy as np
# #import cPickle as pickle
# bbb = np.loadtxt('extra/extra0exta.txt')
# labley=np.zeros((50000,10))
# for i in range(0,50000):
#     labley[i][int(bbb[i+50000])]=1
f=open('extra/extra0exta.txt','rb')
train=pickle.load(f,encoding="latin1")

print(train.shape)
f.close()
f=open('extra/e0.txt','wb')
pickle.dump(train, f)
f.close()
#
#
# #d = dict(name='Bob', age=20, score=88)
# #str = pickle.dumps(d)
# f = open('newlabley2.txt', 'wb')
# pickle.dump(bbb, f)
# f.close()
# print("finished")

# f = open('newgray.txt','rb')
# d = pickle.load(f)
# f.close()
# print("finished")
# print(type(d))

# f = open('extrala1exta.txt','rb')
# labley = pickle.load(f,encoding="latin1")