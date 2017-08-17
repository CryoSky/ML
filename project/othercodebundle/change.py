import numpy as np
import pickle
f = open('extrala1exta.txt','rb')
labley = pickle.load(f,encoding="latin1")
#labley = np.loadtxt('extrala1exta.txt')
f.close()
a=np.zeros((len(labley),10));
for i in range(len(labley)):
    b=int(labley[i])
    a[i,b]=1
#np.savetxt("extrala1exta.txt",a)

f = open('extrala1exta.txt', 'wb')
pickle.dump(a, f)
f.close()

