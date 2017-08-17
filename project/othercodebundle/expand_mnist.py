"""expand_mnist.py
~~~~~~~~~~~~~~~~~~

Take the 50,000 MNIST training images, and create an expanded set of
250,000 images, by displacing each training image up, down, left and
right, by one pixel.  Save the resulting file to
../data/mnist_expanded.pkl.gz.

Note that this program is memory intensive, and may not run on small
systems.

"""

from __future__ import print_function

#### Libraries

# Standard library
import pickle
import gzip
import os.path
import random

# Third-party libraries
import numpy as np

print("Expanding the MNIST training set")

if os.path.exists("../data/mnist_expanded.pkl.gz"):
    print("The expanded training set already exists.  Exiting.")
else:

    f = open('train0321.txt', 'rb')
    bbb = pickle.load(f)  # ,encoding="latin1"




    f.close()
    f = open('newlabley2.txt', 'rb')
    labley = pickle.load(f)  # ,encoding="latin1"
    f.close()
    newy=[]
    for y in labley:
        small=[]
        small.append(y)
        small.append(y)
        small.append(y)
        small.append(y)
        small.append(y)

        newy.append(small)

    newy=np.reshape(newy,(-1,10))
    f= open('expandlable.txt', 'wb')
    pickle.dump(newy, f)
    f.close()




    expanded_training_pairs = []
    j = 0 # counter
    for x in bbb:
        expanded_training_pairs.append(x)
        image = np.reshape(x, (-1, 32))
        j += 1
        if j % 1000 == 0: print("Expanding image number", j)
        # iterate over data telling us the details of how to
        # do the displacement
        for d, axis, index_position, index in [
                (2,  0, "first", 0),
                (-2, 0, "first", 30),
                (2,  1, "last",  0),
                (-2, 1, "last",  30)]:
            new_img = np.roll(image, d, axis)
            if index_position == "first": 
                new_img[index, :] = np.zeros(32)
            else: 
                new_img[:, index] = np.zeros(32)
            expanded_training_pairs.append(np.reshape(new_img, 32*32))

    expanded_training_pairs=np.reshape(expanded_training_pairs,(-1,1024))
    #random.shuffle(expanded_training_pairs)

    print("Saving expanded data. This may take a few minutes.")
    f = open("expand0301.txt", "wb")
    pickle.dump(expanded_training_pairs, f)
    f.close()
    print("ok")