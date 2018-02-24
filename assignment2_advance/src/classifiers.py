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


    #Create LogK to reduce numerical instability
    #axis=1 take the Max for each line
    #Keep the same dimension as logits
    logK = -np.max(logits, axis=1, keepdims = True)

    logitsStable = logits + logK

    expLogitsStable = np.exp(logitsStable)

    #Sum each term of the matrix result from logitsStable
    sumExpLogitsStable = np.sum(expLogitsStable,axis = 1, keepdims = True)

    # Get normalisedProbability of all the classes for each example.
    normalisedProb = expLogitsStable / sumExpLogitsStable

    # Get the number of examples.
    N = logits.shape[0]

    # Get each probability for each class assigned
    #Create a new Matrix according to each class
    logProb = -np.log(normalisedProb[range(N), y])

    #Calculation of loss
    loss = np.sum(logProb) / N

    # Get the gradients of the logits from the formula:
    # dLi / dfk = pk - 1(yi = k).
    dlogits = normalisedProb
    dlogits[range(N), y] -= 1
    dlogits /= N
    


    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
    return loss, dlogits
