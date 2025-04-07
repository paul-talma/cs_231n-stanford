import numpy as np


def softmax(y_hat):
    """
    Computes the softmax vector using the log-sum-exp trick.

    Inputs:
    - y_hat: A numpy array of shape (N, C) containing scores for each class

    Returns:
    - A numpy array of shape (N, C) containing the softmax probabilities for each class
    """
    scaled_y_hat = y_hat - np.max(y_hat, axis=1, keepdims=True)
    return np.exp(scaled_y_hat) / np.sum(np.exp(scaled_y_hat), axis=1, keepdims=True)


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

    N = X.shape[0]
    y_hat = X @ W

    # y_hat_true = y_hat[np.arange(N), y][:, np.newaxis]
    probabilities = softmax(y_hat)
    for i in range(N):
        loss -= np.log(probabilities[i, y[i]])
        for j in range(W.shape[1]):
            dW[:, j] += probabilities[i, j] * X[i]
        dW[:, y[i]] -= X[i]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    loss /= N
    loss += reg * np.sum(W * W)

    dW /= N
    dW += 2 * reg * W
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

    N = X.shape[0]
    y_hat = X @ W

    probabilities = softmax(y_hat)
    loss -= np.sum(np.log(probabilities[np.arange(N), y])) / N + reg * np.sum(W * W)

    probabilities[np.arange(N), y] -= 1
    dW = (X.T @ probabilities) / N + 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
