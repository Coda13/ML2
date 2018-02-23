import numpy as np


def linear_forward(X, W, b):
    """
    Computes the forward pass for a linear (fully-connected) layer.

    The input X has shape (N, d_1, ..., d_K) and contains N samples with each
    example X[i] has shape (d_1, ..., d_K). Each input is reshaped to a
    vector of dimension D = d_1 * ... * d_K and then transformed to an output
    vector of dimension M.

    Args:
    - X: A numpy array of shape (N, d_1, ..., d_K) incoming data
    - W: A numpy array of shape (D, M) of weights, with D = d_1 * ... * d_K
    - b: A numpy array of shape (M, ) of biases

    Returns:
    - out: linear transformation to the incoming data
    """
    # out = None
    """
    TODO: Implement the linear forward pass. Store your result in `out`.
    Tip: Think how X needs to be reshaped.
    """
    ###########################################################################
    #                           BEGIN OF YOUR CODE                            #
    ###########################################################################


    #Linear Forward: Y=WX+b

    #Dimension
    M = len(W[0])

    #Initialisation
    out = []

    #Reshaping X such as each X[i] is of dimension D
    #Using -1 allows to infer the D from N = X.shape[0]
    #We then have a flattend X in which each xi is of dimension D
    X_reshaped = X.reshape(X.shape[0], -1)

    #Output is then equal to the matricial product: X_reshaped * W + b
    #Computing X_reshaped * W
    out = np.matmul(X_reshaped,W)

    #Computing the add of the bias
    for i in range(len(out)):
        for j in range(len(b)):
            out[i][j] += b[j]
        


    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
    return out


def linear_backward(dout, X, W, b):
    """
    Computes the backward pass for a linear (fully-connected) layer.

    Args:
    - dout: Upstream derivative, of shape (N, M)
    - X: A numpy array of shape (N, d_1, ..., d_K) incoming data
    - W: Anumpy array of shape (D, M) of weights, with D= d_1 * ... * d_K
    - b: A numpy array of shape (M, ) of biases

    Returns (as tuple):
    - dX: A numpy array of shape (N, d_1, ..., d_K), gradient with respect to x
    - dW: A numpy array of shape (D, M), gradient with respect to W
    - db: A numpy array of shape (M,), gradient with respect to b
    """
    dX, dW, db = None, None, None
    """
    TODO: Implement the linear backward pass. Store your results of the
    gradients in `dX`, `dW`, `db`.
    """
    ###########################################################################
    #                           BEGIN OF YOUR CODE                            #
    ###########################################################################

    #Dimension
    N = X.shape[0] #or N = len(X[0])
    D = np.prod(X.shape[1:]) #Give the product for d1 to dk

    #Reshape
    X_reshape = np.reshape(X, (N,D))

    #We want a N,D Matrix and dout use N,M so we need a M,D matrix
    dX_reshape = np.matmul(dout, W.T)

    #We want a D,M matrix using a N,M matrix. We need a D,N matrix
    dW = np.matmul(X_reshape.T, dout)

    #We want a M,1 matrix using a N,M Matrix so we need a N,1 matrix
    db = np.matmul(dout.T, np.ones(N)) #Create a matrix N,1 of "1"

    #Now we need to recreate dx according to the dimension given
    # X.shape give the initial dimension
    dX = np.reshape(dX_reshape, X.shape)



    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################

    return dX, dW, db


def relu_forward(X):
    """
    Computes the forward pass for rectified linear unit (ReLU) layer.
    Args:
    - X: Input, an numpy array of any shape
    Returns:
    - out: An numpy array, same shape as X
    """
    out = None
    """
    TODO: Implement the ReLU forward pass. Store your result in out.
    """
    ###########################################################################
    #                           BEGIN OF YOUR CODE                            #
    ###########################################################################

    
    
    out = X.copy()  # Must use copy in numpy to avoid pass by reference.
    out[out < 0] = 0

    out2 = np.maximum(0, out)

    out = out2

    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
    return out


def relu_backward(dout, X):
    """
    Computes the backward pass for rectified linear unit (ReLU) layer.
    Args:
    - dout: Upstream derivative, an numpy array of any shape
    - X: Input, an numpy array with the same shape as dout
    Returns:
    - dX: A numpy array, derivative with respect to X
    """
    dX = None
    """
    TODO: Implement the ReLU backward pass. Store your result in out.
    """
    ###########################################################################
    #                           BEGIN OF YOUR CODE                            #
    ###########################################################################

    #dX must have the same shape as dout and derivative respect to x

    #First we copy dout to dX 
    dX = np.array(dout)

    #Then if X <= 0, element becomes 0
    dX[X<=0] = 0


    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
    return dX


def dropout_forward(X, p=0.5, train=True, seed=42):
    """
    Compute f
    Args:
    - X: Input data, a numpy array of any shape.
    - p: Dropout parameter. We drop each neuron output with probability p.
    Default is p=0.5.
    - train: Mode of dropout. If train is True, then perform dropout;
      Otherwise train is False (= test mode). Default is train=True.

    Returns (as a tuple):
    - out: Output of dropout applied to X, same shape as X.
    - mask: In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    out = None
    mask = None
    if seed:
        np.random.seed(seed)
    """
    TODO: Implement the inverted dropout forward pass. Make sure to consider
    both train and test case. Pay attention scaling the activation function.
    """
    ###########################################################################
    #                           BEGIN OF YOUR CODE                            #
    ###########################################################################

    if train == True:

        #In forward propagation, inputs are set to zero with probability p,
        #and otherwise scaled up by 1/1−p
        
        #We need to generate a 1D random number applicable in all line of X,
        #if this number is >= p then the input is 0

        # The first parameter is th enumber of repetition for 1 element
        # 1-p is the probaility of sucess
        # X.shape is the size wanted
        
        mask = np.random.binomial(1,1-p,X.shape) /(1-p)
    
        
        #Same shape as X     
        out  = X * mask

    else:

        out = X


    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
    return out, mask


def dropout_backward(dout, mask, p=0.5, train=True):
    """
    Compute the backward pass for dropout
    Args:
    - dout: Upstream derivative, a numpy array of any shape
    - mask: In training mode, mask is the dropout mask that was used to
      multiply the input; in test mode, mask is None.
    - p: Dropout parameter. We drop each neuron output with probability p.
    Default is p=0.5.
    - train: Mode of dropout. If train is True, then perform dropout;
      Otherwise train is False (= test mode). Default is train=True.

    Returns:
    - dX: A numpy array, derivative with respect to X
    """
    dX = None
    """
    TODO: Implement the inverted backward pass for dropout. Make sure to
    consider both train and test case.
    """
    ###########################################################################
    #                           BEGIN OF YOUR CODE                            #
    ###########################################################################

    if train == True:
        dX = dout * mask 
    else:
        dX = dout 

    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
    return dX
