import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax(x):
    if len(x.shape) == 1:
        # Vector
        x = x - np.max(x)
        logSum = np.log(np.sum(np.exp(x)))
    else:
        c = np.max(x, axis=1)
        c = np.reshape(c, (c.shape[0], 1))
        x = x - c
        logSum = np.reshape(np.log(np.sum(np.exp(x), axis=1)), (x.shape[0], 1))
    x = np.exp(x - logSum)
    return x


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
    num_train = X.shape[0]
    num_class = W.shape[1]
    for i in range(num_train):
        score = softmax(np.dot(X[i], W))
        loss += -np.log(score[y[i]])
        for j in xrange(num_class):
            if j == y[i]:
                dW[:, j] += (score[j] - 1) * X[i]
            else:
                dW[:, j] += score[j] * X[i]

    loss /= num_train
    loss += reg * np.sum(W ** 2)
    dW = dW / num_train + reg * W

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    num_train = X.shape[0]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    scores = softmax(np.dot(X, W))                            # N by C
    correct_scores = scores[np.arange(len(scores)), y]        # N by 1
    loss += np.sum(-np.log(correct_scores)) / num_train
    loss += reg * np.sum(W ** 2)

    scores[np.arange(len(scores)), y] -= 1                    # N by C
    dW += np.dot(X.T, scores) / num_train + reg * W           # D by C

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
