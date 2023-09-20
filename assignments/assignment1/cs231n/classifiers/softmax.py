from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_classes = W.shape[1]
    num_train = X.shape[0]
    S = X.dot(W)
    U = np.exp(S)
    c = np.sum(U, axis=1) # each training data: sum across all classes.
    for i in range(num_train):
        p = U[i] / c[i] # exp(f_i) / sum(exp(f_k)) (3.4.9 of d2l book)
        loss -= np.log(p[y[i]]) # 3.4.9 of d2l book
        for j in range(num_classes):
            dW[:, j] += X[i] * p[j] # for all the classes
        dW[:, y[i]] -= X[i] # for the correct class. (3.4.9 of d2l book)
    loss /= num_train
    dW /= num_train
    # regularization
    loss += reg * np.sum(np.power(W, 2))
    dW += reg * 2 * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]
    # please refer to the naive version for explanation.
    S = X.dot(W)
    U = np.exp(S)
    c = np.sum(U, axis=1).reshape(-1, 1) # sum(exp(f_k)) reshaped to (num_train, 1)
    p = U / c # exp(f_j) / sum(exp(f_k))
    loss -= np.sum(np.log(p[np.arange(num_train), y]))
    p[np.arange(num_train), y] -= 1
    dW += (X.T).dot(p)
    loss /= num_train
    dW /= num_train
    loss += reg * np.sum(np.power(W, 2))
    dW += reg * 2 * W
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
