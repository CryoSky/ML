import csv
import pickle
import numpy as np
testreader = csv.reader(open("train.csv", "r"))
fxxx = np.zeros((73257, 10), dtype=float)
xxx = list(testreader)
print (len(xxx))
del xxx[0]
del xxx[-1]


for i in range(len(xxx)):
    if int(xxx[i][0])==10:
        fxxx[i][0]=1
    else:
        fxxx[i][int(xxx[i][0])]=1;
print(len(fxxx))
f = open('newlabley2.txt', 'wb')
pickle.dump(fxxx, f)
f.close()
