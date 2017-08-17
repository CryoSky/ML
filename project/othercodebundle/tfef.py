import numpy as np
# import csv
import pickle
# # testreader = csv.reader(open("test.csv", "r"))
# # fxxx=np.zeros((26032,3072),dtype=float)
# # xxx = list(testreader)
# # xxx = np.array(xxx[1:]).astype("float32")
# # for i in range(len(xxx)):
# #     fxxx[i] = xxx[i]
# # print (fxxx.shape)
# # fxxx=fxxx.reshape(32, 32,3,-1).transpose(3,2,0,1)
# # fxxx=fxxx[:,0]*0.3+fxxx[:,1]*0.59+fxxx[:,2]*0.11
# # fxxx=fxxx/256
# # fxxx=fxxx.reshape(26032,1024)
# # small=np.zeros([8,1024])
# # fxxx=np.vstack((fxxx,small))
# # print(np.min(fxxx),np.max(fxxx))
# # f = open('test0309.txt', 'wb')
# # pickle.dump(fxxx, f)
# # f.close()
#
# # f = open('1.txt','rb')
# # bbb = pickle.load(f,encoding="latin1")
# # f.close()
# #
# #
# # print(np.max(bbb))
# # f = open('extratring1.txt','wb')
# # pickle.dump(bbb, f)
# # f.close()
#
# labley=np.loadtxt('extrala.txt')
# bbb=labley[0:50000]
# print(bbb.shape)
# matr=np.zeros((50000,10))
# for i in range(50000):
#     matr[i][int(bbb[i])]=1
# print(matr.shape)
# print (matr[0:10,0:10])
# f = open('extralable50000.txt','wb')
# pickle.dump(matr, f)
# f.close()
#
# print("max,min",np.max(bbb),np.min(bbb))
# for i in range(5):
#     for j in range(5):
#         curr = bbb[5 * (i) + j ].reshape(32, 32)
#         curr = curr * 256;
#         plt.subplot(5, 5, 5 * (i) + j + 1)
#         plt.imshow(curr.astype('uint8'))
#
# plt.show()

# a=np.array([1,2,3])
# b=np.array([2,3])
# c=np.stack((a,b),axis=-1)
# print (c)
f=open("the1exta.txt","rb")
bbb=pickle.load(f,encoding="latin1")
f.close()
print(bbb[0])
f=open("secondtxtra.txt","wb")
pickle.dump(bbb,f)
f.close()