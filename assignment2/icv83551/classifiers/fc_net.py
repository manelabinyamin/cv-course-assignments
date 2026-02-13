import os
from builtins import object, range

import numpy as np

from ..layer_utils import *
from ..layers import *


class FullyConnectedNet(object):
    """Class for a multi-layer fully connected neural network.

    Network contains an arbitrary number of hidden layers, ReLU nonlinearities,
    and a softmax loss function. This will also implement dropout and batch/layer
    normalization as options. For a network with L layers, the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional and the {...} block is
    repeated L - 1 times.

    Learnable parameters are stored in the self.params dictionary and will be learned
    using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout_keep_ratio=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout_keep_ratio: Scalar between 0 and 1 giving dropout strength.
            If dropout_keep_ratio=1 then the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
            are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
            initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
            this datatype. float32 is faster but less accurate, so you should use
            float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers.
            This will make the dropout layers deteriminstic so we can gradient check the model.
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        for layer in range(self.num_layers):
            if layer == 0:
                input_size = input_dim
                output_size = hidden_dims[0]
            elif layer == self.num_layers - 1:
                input_size = hidden_dims[-1]
                output_size = num_classes
            else:
                input_size = hidden_dims[layer - 1]
                output_size = hidden_dims[layer]

            if isinstance(weight_scale, list):
                ws = weight_scale[layer]
            else:
                ws = weight_scale
            self.params[f"W{layer + 1}"] = ws * np.random.randn(input_size, output_size)
            self.params[f"b{layer + 1}"] = np.zeros(output_size)

            # Initialize batch normalization parameters for all layers except the last
            if self.normalization == "batchnorm" and layer != self.num_layers - 1:
                self.params[f"gamma{layer + 1}"] = np.ones(output_size)
                self.params[f"beta{layer + 1}"] = np.zeros(output_size)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype.
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """Compute loss and gradient for the fully connected net."""
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode

        ############################################################################
        # Forward pass
        ############################################################################
        scores = None
        h = X
        cache_dict = {}

        for i in range(self.num_layers):
            W = self.params[f"W{i + 1}"]
            b = self.params[f"b{i + 1}"]

            if i != self.num_layers - 1:
                # Hidden layers: affine - [batchnorm] - relu - [dropout]
                if self.normalization == "batchnorm":
                    gamma = self.params[f"gamma{i + 1}"]
                    beta = self.params[f"beta{i + 1}"]
                    h, cache = affine_bn_relu_forward(
                        h, W, b, gamma, beta, self.bn_params[i]
                    )
                    cache_dict[f"cache{i + 1}"] = cache
                else:
                    a, cache_a = affine_forward(h, W, b)
                    h, cache_h = relu_forward(a)
                    cache_dict[f"cache_a{i + 1}"] = cache_a
                    cache_dict[f"cache_h{i + 1}"] = cache_h

                # Dropout (if enabled)
                if self.use_dropout:
                    h, cache_dropout = dropout_forward(h, self.dropout_param)
                    cache_dict[f"cache_dropout{i + 1}"] = cache_dropout
            else:
                # Last layer: just affine (no normalization, no ReLU)
                scores, cache_a = affine_forward(h, W, b)
                cache_dict[f"cache_a{i + 1}"] = cache_a

        # If test mode return early
        if mode == "test":
            return scores

        ############################################################################
        # Backward pass
        ############################################################################
        loss, grads = 0.0, {}

        # Compute loss
        loss, dscores = softmax_loss(scores, y)

        # Add L2 regularization to loss
        reg_val = sum(
            np.sum(self.params[f"W{i + 1}"] * self.params[f"W{i + 1}"])
            for i in range(self.num_layers)
        )
        loss += 0.5 * self.reg * reg_val

        # Backward pass through layers
        dh = dscores
        for i in reversed(range(self.num_layers)):
            if i != self.num_layers - 1:
                # Backward through dropout
                if self.use_dropout:
                    cache_dropout = cache_dict[f"cache_dropout{i + 1}"]
                    dh = dropout_backward(dh, cache_dropout)

                # Backward through hidden layer
                if self.normalization == "batchnorm":
                    cache = cache_dict[f"cache{i + 1}"]
                    dh, dW, db, dgamma, dbeta = affine_bn_relu_backward(dh, cache)
                    grads[f"gamma{i + 1}"] = dgamma
                    grads[f"beta{i + 1}"] = dbeta
                else:
                    cache_h = cache_dict[f"cache_h{i + 1}"]
                    cache_a = cache_dict[f"cache_a{i + 1}"]
                    dh = relu_backward(dh, cache_h)
                    dh, dW, db = affine_backward(dh, cache_a)
            else:
                # Backward through last layer
                cache_a = cache_dict[f"cache_a{i + 1}"]
                dh, dW, db = affine_backward(dh, cache_a)

            # Store gradients with L2 regularization for weights
            grads[f"W{i + 1}"] = dW + self.reg * self.params[f"W{i + 1}"]
            grads[f"b{i + 1}"] = db

        return loss, grads

    def save(self, fname):
        """Save model parameters."""
        fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
        params = self.params
        np.save(fpath, params)
        print(fname, "saved.")

    def load(self, fname):
        """Load model parameters."""
        fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
        if not os.path.exists(fpath):
            print(fname, "not available.")
            return False
        else:
            params = np.load(fpath, allow_pickle=True).item()
            self.params = params
            print(fname, "loaded.")
            return True
