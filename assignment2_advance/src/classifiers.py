import numpy as np


def softmax(logits, y):
    """
    Computes the loss and gradient for softmax classification.

    Args:
    - logits: A numpy array of shape (N, C)
    - y: A numpy array of shape (N,). y represents the labels corresponding to
    logits, where y[i] is the label of logits[i], and the value of y have a 
    range of 0 <= y[i] < C

    Returns (as a tuple):
    - loss: Loss scalar
    - dlogits: Loss gradient with respect to logits
    """
    loss, dlogits = None, None
    """
    TODO: Compute the softmax loss and its gradient using no explicit loops
    Store the loss in loss and the gradient in dW. If you are not careful
    here, it is easy to run into numeric instability. Don't forget the
    regularization!
    """
    ###########################################################################
    #                           BEGIN OF YOUR CODE                            #
    ###########################################################################

    # Get the number of examples.
    n = logits.shape[0]

    # Set up variable logC to reduce numerical instability.
    logC = -np.max(logits, axis=1, keepdims = True)
    logitsReduced = logits + logC
    expLogitsReduced = np.exp(logitsReduced)
    sumExpLogitsReduced = np.sum(expLogitsReduced, axis = 1, keepdims = True)

    # Get normalisedProbability of all the classes for each example.
    normProbabilities = expLogitsReduced / sumExpLogitsReduced

    # Get the probabilities assigned to the correct class for each example,
    # and then compute the loss (no regularisation needs to occur).
    correctLogProbabilities = -np.log(normProbabilities[range(n), y])
    loss = np.sum(correctLogProbabilities) / n

    # Get the gradients of the logits from the formula:
    # dLi / dfk = pk - 1(yi = k).
    dlogits = normProbabilities
    dlogits[range(n), y] -= 1
    dlogits /= n
    


    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
    return loss, dlogits
