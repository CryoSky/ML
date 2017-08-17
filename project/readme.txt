Guangyuan Yu, Andrew wells
GY12.AW34
Python3, TENSORFLOW


I run my code under the anadonda with python3 on a windows platform with GPU.  
My anaconda navigator has two environment. On is the root(python2), where I open the jupyter to finish the homework, another is python3, which has tensorflow. I use the spyder. 




I suppose you have anadonda2. And I test these lines tonight. First, creat a environment called py80 under anaconda2. You can change the name to anything else. These code is for windows.


conda create -n py80 python=3.5
activate tensorflow


Here, the code of windows platform is one line. 
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/windows/gpu/tensorflow_gpu-1.0.1-cp35-cp35m-win_amd64.whl



For mac, I haven't test the code. It is just a copy from tensorflow website. I think you should add python=35 in the first line, I am not sure.  
$ conda create -n tensorflow
$ source activate tensorflow
$ pip install --ignore-installed --upgrade $TF_PYTHON_URL 
$ pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.0.1-py3-none-any.whl


How to run my code?
Open andaconda navigator. Select environment, click a button to change the "root" to the py80. Click on the spyder. click on project, click on open project. click on existing directory. Click on the folder which contains the .py file. Click on creat. Please copy all the three data set near the .py file.



1. Run from scratch: First, run the dataprepossing file to generate the grey file we need. Then run the finalproject file, make sure you comment the saver.restore() function.
Change the learning rate by hand. The spyder enable to use GPU and stop with all the parameters in the memory. So you can stop and store the model by selecting the line of "saver.save(sess, "codebackup/model7.ckpt")". You have to close and restart the spyder again to load the parameter. 

2. Reproducet the result. Make use uncomment the saver.resore() function, and make sure the traing accuracy is about 97%. Maybe the training accuracy is just about 12%, it means the spyder fails to load the saved model, so you need to stop and quit spyder, and open spyder to run again. 
3. I handle the dataset on my mac. It is because the extraset have problem on windows platform.





from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pickle
import tensorflow as tf