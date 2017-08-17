# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.

See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
FLAGS = None


def main(_):
  # Import data
 # mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
  print ("loading bbb")
  bbb = np.loadtxt('gray.txt')  # 0 to1
  print ("loading lable")
  labley = np.loadtxt('newlabley.txt')
  #testreader = csv.reader(open("test.csv", "rb"))
  #fxxx = np.loadtxt('testgray.txt')
  #small = np.zeros([8, 1024])
  #fxxx = np.vstack((fxxx, small))
  #n = 10000
  #dx = 10000



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
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  # Train
  for i in range(500):
    batch_xs, batch_ys =bbb[100*i:100*i+100],labley[100*i:100*i+100] #mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(sess.run(accuracy, feed_dict={x: bbb[60000:61000],#mnist.test.images,
                                      y_: labley[60000:61000]}))









if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  #parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      #help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
