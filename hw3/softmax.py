import numpy as np
from random import shuffle
import scipy.sparse

def softmax_loss_naive(theta, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)
  Inputs:
  - theta: d x K parameter matrix. Each column is a coefficient vector for class k
  - X: m x d array of data. Data are d-dimensional rows.
  - y: 1-dimensional array of length m with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to parameter matrix theta, an array of same size as theta
  """
  # Initialize the loss and gradient to zero.

  J = 0.0
  grad = np.zeros_like(theta)
  m, dim = X.shape
  # theta.shape
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in J and the gradient in grad. If you are not              #
  # careful here, it is easy to run into numeric instability. Don't forget    #
  # the regularization term!                                                  #
  #############################################################################

  gradtemp=np.zeros_like(theta.T)
  k=max(y)

  do=0.0
  for j in range(0,m):
   
    do=sum(np.exp(X[j].dot(theta)))
    so=np.exp(X[j].dot(theta.T[y[j]]))
  
    J-= np.log(so/do)
    for i in range(0,k+1):
      so2=np.exp(X[j].dot(theta.T[i]))
      #print so2.shape
      gradtemp[i]-=[-so2/do]*X[j]
      if y[j]==i:
        gradtemp[i]-=[1.0]*X[j]

  J=(J+reg/2*np.sum(np.square(theta)))/m
  grad=(gradtemp.T+reg*theta)/m








  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return J, grad

  
def softmax_loss_vectorized(theta, X, y, reg):
  """
  Softmax loss function, vectorized version.
  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.

  J = 0.0
  grad = np.zeros_like(theta)
  #print grad.shape
  m, dim = X.shape

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in J and the gradient in grad. If you are not careful      #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization term!                                                      #
  #############################################################################
  # xt=X.dot(theta)
  # Pt=np.exp(xt-np.max(xt,1).reshape([m,1]).dot(np.ones([1,theta.shape[1]])))
  # P=Pt/Pt.sum(1).reshape([m,1]).dot(np.ones([1,theta.shape[1]]))
  # print P.shape
  # print y.shape
  ## J=-1.0/m*np.sum(np.multiply(np.log(P),convert_y_to_matrix(y)))+(reg/2/m)*np.sum(theta**2)
  # grad=-1.0/m*X.T.dot((convert_y_to_matrix(y)-P))+ theta*reg/m
  # scores = np.dot(X,theta)
  # scores -= np.max(scores) # shift by log C to avoid numerical instability
  # (C, D) = theta.shape
  # print scores.shape
  # y_mat = np.zeros(shape = (C, m))
  # print "mat", y_mat.shape
  # y_mat[y, range(m)] = 1

  # # matrix of all zeros except for a single wx + log C value in each column that corresponds to the
  # # quantity we need to subtract from each row of scores
  # correct_wx = np.dot(y_mat, scores)

  # # create a single row of the correct wx_y + log C values for each data point
  # sums_wy = np.sum(correct_wx, axis=0) # sum over each column

  # exp_scores = np.exp(scores)
  # sums_exp = np.sum(exp_scores, axis=0) # sum over each column
  # result = np.log(sums_exp)

  # result -= sums_wy

  # J = np.sum(result)

  # # Right now the loss is a sum over all training examples, but we want it
  # # to be an average instead so we divide by num_train.
  # J /= float(m)

  # # Add regularization to the loss.
  # J += 0.5 * reg * np.sum(theta * theta)/m

  # sum_exp_scores = np.sum(exp_scores, axis=0) # sum over columns
  # sum_exp_scores = 1.0 / (sum_exp_scores + 1e-8)

  # #grad  = exp_scores * sum_exp_scores
  # # grad  = np.dot(X.T,grad)
  # print grad.shape
  # # grad  -= np.dot(y_mat, X.T)

  # grad  /= float(m)

  # Add regularization to the gradient
  #grad  += reg * theta
  num_train = X.shape[0]
  # Xnew=np.zeros_like(X)
  # for i in range(num_train):
  #   Xnew[i]=X[i]-np.max(X[i])
  
  scores = np.dot(X, theta)
  #49000,10
  #scores = np.dot(Xnew, theta)
  exp_scores = np.exp(scores)
  prob_scores = exp_scores/np.sum(exp_scores, axis=1, keepdims=True)
  correct_log_probs = -np.log(prob_scores[range(num_train), y])
  J = np.sum(correct_log_probs)
  J /= num_train
  J += 0.5 * reg * np.sum(theta**2)/m

  # grads
  dscores = prob_scores
  dscores[range(num_train), y] -= 1.0
  grad = np.dot(X.T, dscores)

  grad /= m
  grad += reg * theta/m
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return J, grad
