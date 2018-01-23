import numpy as np

from layers import *
from fast_layers import *
from layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys      #
    # 'theta1' and 'theta1_0'; use keys 'theta2' and 'theta2_0' for the        #
    # weights and biases of the hidden affine layer, and keys 'theta3' and     #
    # 'theta3_0' for the weights and biases of the output affine layer.        #
    ############################################################################
    # about 12 lines of code

    self.params['theta1'] = np.random.randn(
            num_filters, input_dim[0], filter_size, filter_size) * weight_scale
    self.params['theta1_0'] = np.zeros((num_filters,))
    self.params['theta2'] = np.random.randn(
            num_filters * input_dim[1] * input_dim[2] / 4, hidden_dim) * weight_scale
    self.params['theta2_0'] = np.zeros((hidden_dim,))
    self.params['theta3'] = np.random.randn(
            hidden_dim, num_classes) * weight_scale
    self.params['theta3_0'] = np.zeros((num_classes,))
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    theta1, theta1_0 = self.params['theta1'], self.params['theta1_0']
    theta2, theta2_0 = self.params['theta2'], self.params['theta2_0']
    theta3, theta3_0 = self.params['theta3'], self.params['theta3_0']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = theta1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    # about 3 lines of code (use the helper functions in layer_utils.py)


    out1, cache1 = conv_relu_pool_forward(X, theta1, theta1_0, conv_param, pool_param)
    out2, cache2 = affine_relu_forward(out1, theta2, theta2_0)
    scores, cache3 = affine_forward(out2, theta3, theta3_0)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    # about 12 lines of code

    loss, dxl = softmax_loss(scores, y)
    loss += self.reg / 2. * (np.sum(theta1 ** 2) +
                                 np.sum(theta2 ** 2) +
                                 np.sum(theta3 ** 2))

    dx3, dtheta3, dtheta_03 = affine_backward(dxl, cache3)
    grads['theta3'] = dtheta3 + self.reg * theta3
    grads['theta3_0'] = dtheta_03

    dx2, dtheta2, dtheta_02 = affine_relu_backward(dx3, cache2)
    grads['theta2'] = dtheta2 + self.reg * theta2
    grads['theta2_0'] = dtheta_02

    dx, dtheta, dtheta_01 = conv_relu_pool_backward(dx2, cache1)
    grads['theta1'] = dtheta + self.reg * theta1
    grads['theta1_0'] = dtheta_01
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
class Newmodel(object):
  def __init__(self, input_dim=(3, 32, 32), num_filters1=32, num_filters2 = 16, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0, dtype=np.float32):
    '''
    [conv-relu-pool]-[conv-relu]-   affine-softmax
    theta1           theta2         theta3
    num_filters1     num_filters2

    '''
    self.params = {}
    self.reg = reg
    self.dtype = dtype

    C, H, W = input_dim
    self.params['theta1'] = np.random.normal(0, weight_scale, (num_filters1, C, filter_size, filter_size))
    self.params['theta1_0'] = np.zeros(num_filters1)
    self.params['theta2'] = np.random.normal(0,weight_scale,(num_filters2,num_filters1,filter_size,filter_size))
    self.params['theta2_0'] = np.zeros(num_filters2)
    self.params['theta3'] = np.random.normal(0, weight_scale, (num_filters2 * H * W / 4, hidden_dim))
    self.params['theta3_0'] = np.zeros(hidden_dim)
    self.params['theta4'] = np.random.normal(0,weight_scale,(hidden_dim,num_classes))
    self.params['theta4_0'] = np.zeros(num_classes)

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)

  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.

    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    theta1, theta1_0 = self.params['theta1'], self.params['theta1_0']
    theta2, theta2_0 = self.params['theta2'], self.params['theta2_0']
    theta3, theta3_0 = self.params['theta3'], self.params['theta3_0']
    theta4, theta4_0 = self.params['theta4'], self.params['theta4_0']

    # pass conv_param to the forward pass for the convolutional layer
    filter_size = theta1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}


    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    out1, cache1 = conv_relu_pool_forward(X, theta1, theta1_0, conv_param, pool_param)
    out2, cache2 = conv_relu_forward(out1, theta2, theta2_0, conv_param)
    out3, cache3 = affine_relu_forward(out2, theta3, theta3_0)
    scores, cache4 = affine_forward(out3, theta4, theta4_0)

    
    if y is None:
      return scores

    loss, grads = 0, {}

    loss, dx = softmax_loss(scores, y)
    loss += 0.5 * self.reg * (np.sum(theta1 ** 2) + np.sum(theta2 ** 2) + np.sum(theta3 ** 2) + np.sum(theta4**2))
    dx4, dtheta4, dtheta4_0 = affine_backward(dx, cache4)
    grads['theta4'] = dtheta4+ self.reg*theta4
    grads['theta4_0'] = dtheta4_0
    dx3, dtheta3, dtheta3_0 = affine_relu_backward(dx4, cache3)
    grads['theta3'] = dtheta3 + self.reg * theta3
    grads['theta3_0'] = dtheta3_0
    dx2, dtheta2, dtheta2_0 = conv_relu_backward(dx3, cache2)
    grads['theta2'] = dtheta2 + self.reg * theta2
    grads['theta2_0'] = dtheta2_0
    dx1, dtheta1, dtheta1_0 = conv_relu_pool_backward(dx2, cache1)
    grads['theta1'] = dtheta1 + self.reg * theta1
    grads['theta1_0'] = dtheta1_0

    return loss, grads  
  

