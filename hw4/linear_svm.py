import numpy as np

##################################################################################
#   Two class or binary SVM                                                      #
##################################################################################

def binary_svm_loss(theta, X, y, C):
  """
  SVM hinge loss function for two class problem

  Inputs:
  - theta: A numpy vector of size d containing coefficients.
  - X: A numpy array of shape mxd 
  - y: A numpy array of shape (m,) containing training labels; +1, -1
  - C: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to theta; an array of same shape as theta
"""

  m, d = X.shape
  grad = np.zeros(theta.shape)
  J = 0

  ############################################################################
  # TODO                                                                     #
  # Implement the binary SVM hinge loss function here                        #
  # 4 - 5 lines of vectorized code expected                                  #
  ############################################################################
  correctness = y * X.dot(theta)
  J = 1. / (2 * m) * sum(theta ** 2) + 1. * C / m * sum(1 - correctness[correctness < 1])
  grad = 1. / m * theta + 1. * C / m * (-y[correctness < 1].dot(X[correctness < 1]))
  return J,grad
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  

##################################################################################
#   Multiclass SVM                                                               #
##################################################################################

# SVM multiclass

def svm_loss_naive(theta, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension d, there are K classes, and we operate on minibatches
  of m examples.

  Inputs:
  - theta: A numpy array of shape d X K containing parameters.
  - X: A numpy array of shape m X d containing a minibatch of data.
  - y: A numpy array of shape (m,) containing training labels; y[i] = k means
    that X[i] has label k, where 0 <= k < K.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss J as single float
  - gradient with respect to weights theta; an array of same shape as theta
  """

  delta = 1.0
  dtheta = np.zeros(theta.shape) # initialize the gradient as zero

  # compute the loss function

  K = theta.shape[1]
  m = X.shape[0]
  dtheta = np.zeros(theta.shape)
  J = 0.0
  for i in xrange(m):
    scores = X[i,:].dot(theta)
    correct_class_score = scores[y[i]]
    for j in xrange(K):
      if j == y[i]:
        continue
      margin = max(0,scores[j] - correct_class_score + delta)
      #J += margin
      if margin > 0:
        J += margin
        dtheta[:,j] += X[i]
        dtheta[:,y[i]] -= X[i] 
    #pass

  # Right now the loss is a sum over all training examples, but we want it
  # To be an average instead so we divide by num_train.
  J /= m
  dtheta /=m
  # Add regularization to the loss.
  J += 0.5 * reg * np.sum(theta * theta)/m

  dtheta +=reg*theta/m


  


  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dtheta.            #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return J, dtheta


def svm_loss_vectorized(theta, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  J = 0.0
  dtheta = np.zeros(theta.shape) # initialize the gradient as zero
  delta = 1.0
  num_classes = theta.shape[1]
  num_train = X.shape[0]
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in variable J.                                                     #
  #############################################################################
  scores = X.dot(theta)
  Y = np.zeros(scores.shape)
  for i,row in enumerate(Y):
    row[y[i]] = 1
  
  Correct_class_scores = np.array( [ [ scores[i][y[i]] ]*num_classes for i in range(num_train) ] )  
  Margin = scores - Correct_class_scores + ((scores - Correct_class_scores) != 0)*delta #matrix
  X_with_margin_count = np.multiply(X.T , ( Margin > 0).sum(1) ).T

  J += np.sum((Margin>0)*Margin)/num_train
  J += 0.5 * reg * np.sum(theta * theta)/num_train
  dtheta += ( Margin > 0 ).T.dot(X).T/num_train
  dtheta -= (Margin == 0).T.dot(X_with_margin_count).T/num_train
  dtheta += reg*theta/num_train


  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dtheta.                                       #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return J, dtheta
