from sklearn import linear_model
import numpy as np
import utils

class one_vs_allLogisticRegressor:

    def __init__(self,labels):
        self.theta = None
        self.labels = labels

    def train(self,X,y,reg):
        
        """
        Use sklearn LogisticRegression for training K classifiers in one-vs-rest mode
        Read the documentation carefully and choose an appropriate solver. Choose
        the L2 penalty. Remember that the X data has a column of ones prepended to it.
        Set the appropriate flag in logisticRegression to cover that.
        
        X = m X (d+1) array of training data. Assumes X has an intercept column
        y = 1 dimensional vector of length m (with K labels)
        reg = regularization strength

        Computes coefficents for K classifiers: a matrix with (d+1) rows and K columns
           - one theta of length d for each class
       """
        
        m,dim = X.shape
        theta_opt = np.zeros((dim,len(self.labels)))

        ###########################################################################
        # Compute the K logistic regression classifiers                           #
        # TODO: 7-9 lines of code expected                                        #
        ###########################################################################
        k=len(self.labels);
        ytemp=[0 for i in range(0,len(y))];
        theta_temp = np.zeros((len(self.labels),dim))
        for i in range(0,k):
            for j in range(0, len(y)):
                if y[j] == i:
                    ytemp[j] = 1
                else:
                    ytemp[j] = 0

            clf=linear_model.LogisticRegression(C=1e5,solver='lbfgs',fit_intercept=False)
            print "here"
            print X.shape
            
            clf.fit(X, ytemp)
            print "fit"
            print "haha",clf.coef_.T
            theta_temp[i]=clf.coef_
            print "yes",theta_temp[i]
        theta_opt=  theta_temp.T

        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        self.theta = theta_opt

    def predict(self,X):
        """
        Use the trained weights of this linear classifier to predict labels for'l2'        data points.

        Inputs:
        - X: m x (d+1) array of training data. 

        Returns:
        - y_pred: Predicted output for the data in X. y_pred is a 1-dimensional
          array of length m, and each element is a class label from one of the
          set of labels -- the one with the highest probability
        """

        y_pred = np.zeros(X.shape[0])


        ###########################################################################
        # Compute the predicted outputs for X                                     #
        # TODO: 1-2 lines of code expected                                        #
        ###########################################################################
        b=X.dot(self.theta)
        print b.shape
        for j in range (0,b.shape[0]):
            y_pred[j]=b[j].argmax(axis=0) # there is problem here
        print "ok"
            ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return y_pred

