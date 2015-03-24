# Credits to Kyle Kastner
import numpy as np
import theano.tensor as T
from theano.tensor.signal.downsample import max_pool_2d
from utils import shared_normal, shared_zeros


def _relu(x):
    return x * (x > 1E-6)


def conv_layer(input_variable, filter_shape, pool_shape, stride, rng):
    filters = shared_normal(rng, filter_shape)
    biases = shared_zeros(filter_shape[0])
    params = [filters, biases]
    # subsample=stride : this is the distance between the receptive field
    # centers of neighboring neurons in a kernel map (It's not pooling!)
    conv = T.nnet.conv2d(input_variable, filters, subsample=stride) + \
        biases.dimshuffle('x', 0, 'x', 'x')
    out = _relu(conv)
    pooled = max_pool_2d(out, pool_shape, ignore_border=True)
    return pooled, params


def fc_layer(input_variable, layer_shape, rng):
    w = shared_normal(rng, layer_shape)
    b = shared_zeros(layer_shape[1])
    params = [w, b]
    out = _relu(T.dot(input_variable, w) + b)
    return out, params


def softmax_layer(input_variable, layer_shape, rng):
    w = shared_normal(rng, layer_shape)
    b = shared_zeros(layer_shape[1])
    params = [w, b]
    out = T.dot(input_variable, w) + b
    e = T.exp(out - out.max(axis=1, keepdims=True))
    softmax = e / e.sum(axis=1, keepdims=True)
    # Gradient of softmax not defined... again!
    # softmax = T.nnet.softmax(out)
    return softmax, params


def cross_entropy(y_hat_sym, y_sym):
    return -T.mean(
        T.log(y_hat_sym)[T.arange(y_sym.shape[0]), y_sym])


def minibatch_indices(x, minibatch_size, lb=None, ub=None):
    if lb is None:
        lb = 0
    if ub is None:
        ub = len(x)
    minibatch_indices = np.arange(lb, ub, minibatch_size)
    minibatch_indices = np.asarray(list(minibatch_indices) + [ub])
    start_indices = minibatch_indices[:-1]
    end_indices = minibatch_indices[1:]
    return zip(start_indices, end_indices)
