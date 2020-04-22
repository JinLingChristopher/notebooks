from builtins import object

from cs231n.layer_utils import *


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


def affine_bn_relu_forward(x, w, b, gamma, beta, bn_param):
    linearOut, linearCache = affine_forward(x, w, b)
    bnOut, bnCache = batchnorm_forward(linearOut, gamma, beta, bn_param)
    reluOut, reluCache = relu_forward(bnOut)

    cache = linearCache, bnCache, reluCache
    return reluOut, cache


def affine_bn_relu_backword(dout, cache):
    linearCache, bnCache, reluCache = cache
    da = relu_backward(dout, reluCache)
    dz, dgamma, dbeta = batchnorm_backward(da, bnCache)
    dx, dw, db = affine_backward(dz, linearCache)

    return dx, dw, db, dgamma, dbeta


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian with standard deviation equal to   #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with first layer  #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases using the keys 'W2' and 'b2'.                                 #
        ############################################################################
        self.params['W1'] = np.random.randn(input_dim, hidden_dim) * weight_scale
        self.params['b1'] = np.zeros((hidden_dim, ))
        self.params['W2'] = np.random.randn(hidden_dim, num_classes) * weight_scale
        self.params['b2'] = np.zeros((num_classes, ))
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        firstOut, firstCache = affine_forward(X, self.params['W1'], self.params['b1'])
        reluOut, reluCache = relu_forward(firstOut)
        scores, secondCache = affine_forward(reluOut, self.params['W2'], self.params['b2'])
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        num_train = len(X)
        probs = softmax(scores)
        dataLoss = np.sum(-np.log(probs[np.arange(len(probs)), y])) / num_train
        regLoss = 0.5 * self.reg * (np.sum(self.params['W1'] ** 2) + np.sum(self.params['W2'] ** 2))
        loss = dataLoss + regLoss

        # computer derivatives
        delta2 = probs
        delta2[np.arange(len(delta2)), y] -= 1

        dx2, dw2, db2 = affine_backward(delta2, secondCache)
        dw2 = dw2 / num_train + self.reg * self.params['W2']
        db2 = db2 / num_train

        dout = np.dot(delta2, self.params['W2'].T)
        delta1 = relu_backward(dout, reluOut)
        dx1, dw1, db1 = affine_backward(delta1, firstCache)
        dw1 = dw1 / num_train + self.reg * self.params['W1']
        db1 = db1 / num_train

        grads['W2'] = dw2
        grads['b2'] = db2
        grads['W1'] = dw1
        grads['b1'] = db1

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution with standard deviation equal to  #
        # weight_scale and biases should be initialized to zero.                   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to one and shift      #
        # parameters should be initialized to zero.                                #
        ############################################################################
        dimensions = [input_dim] + hidden_dims + [num_classes]
        L = len(dimensions)
        for l in range(1, L):
            self.params['W' + str(l)] = np.random.randn(dimensions[l-1], dimensions[l]) * weight_scale
            self.params['b' + str(l)] = np.zeros((dimensions[l], ))
            if self.use_batchnorm and l != L-1:
                self.params['gamma' + str(l)] = np.ones(dimensions[l])
                self.params['beta' + str(l)] = np.zeros(dimensions[l])
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.
        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # forward through each layer without the last one.
        A_prev = X
        caches = []
        for l in range(1, self.num_layers):
            weight, bias = self.params['W' + str(l)], self.params['b' + str(l)]
            if self.use_batchnorm:
                gamma, beta = self.params['gamma' + str(l)], self.params['beta' + str(l)]
                activation, cache = affine_bn_relu_forward(A_prev, weight, bias, gamma, beta, self.bn_params[l-1])
            else:
                activation, cache = affine_relu_forward(A_prev, weight, bias)
            if self.use_dropout:
                activation, dropoutCache = dropout_forward(activation, self.dropout_param)
                cache = cache, dropoutCache
            A_prev = activation
            caches.append(cache)
        # forward to the last output layer
        weight = self.params['W' + str(self.num_layers)]
        bias = self.params['b' + str(self.num_layers)]
        scores, cache = affine_forward(A_prev, weight, bias)
        caches.append(cache)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################

        # Compute the loss the output layer
        num_train = len(X)
        # probs = softmax(scores)
        # dataLoss = np.sum(np.nan_to_num(-np.log(probs[np.arange(len(probs)), y]))) / num_train
        # weights = [self.params['W' + str(l)] for l in range(1, self.num_layers + 1)]
        # regLoss = 0.5 * self.reg * np.sum([np.sum(w**2) for w in weights])
        # loss = dataLoss + regLoss

        dataLoss, dOut = softmax_loss(scores, y)
        weights = [self.params['W' + str(l)] for l in range(1, self.num_layers + 1)]
        regLoss = 0.5 * self.reg * np.sum([np.sum(w ** 2) for w in weights])
        loss = dataLoss + regLoss
        # Compute derivatives of the output layer
        # dOut = probs
        # dOut[np.arange(len(dOut)), y] -= 1
        linear_cache = caches[self.num_layers - 1]
        dA_temp, dw_temp, db_temp = affine_backward(dOut, linear_cache)
        grads['W' + str(self.num_layers)] = \
            dw_temp + self.reg * self.params['W' + str(self.num_layers)]
        grads['b' + str(self.num_layers)] = db_temp

        # Compute backward derivatives of each hidden layer
        for l in reversed(range(1, self.num_layers)):
            dOut = dA_temp
            cache = caches[l - 1]
            if self.use_dropout:
                cache, dropoutCache = cache
                dOut = dropout_backward(dOut, dropoutCache)
            if self.use_batchnorm:
                dA_temp, dw_temp, db_temp, dgamma_temp, dbeta_temp = affine_bn_relu_backword(dOut, cache)
                grads['gamma' + str(l)] = dgamma_temp
                grads['beta' + str(l)] = dbeta_temp
            else:
                dA_temp, dw_temp, db_temp = affine_relu_backward(dOut, cache)
            grads['W' + str(l)] = dw_temp + self.reg * self.params['W' + str(l)]
            grads['b' + str(l)] = db_temp
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads