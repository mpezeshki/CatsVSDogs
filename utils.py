import numpy as np
import theano
import theano.tensor as T


def shared_uniform(rng, shape, low=-0.5, high=1.5, name=None):
    v = np.asarray(rng.uniform(size=shape, low=low, high=high),
                   dtype=theano.config.floatX)
    return theano.shared(v, name=name)


def shared_normal(rng, shape, mean=0.0, stdev=0.25, name=None):
    v = np.asarray(mean + rng.standard_normal(shape) * stdev,
                   dtype=theano.config.floatX)
    return theano.shared(v, name=name)


def shared_zeros(shape, name=None):
    v = np.zeros(shape, dtype=theano.config.floatX)
    return theano.shared(v, name=name)


def logsumexp(x, dim=-1):
    """Compute log(sum(exp(x))) in a numerically stable way.

       Use second argument to specify along which dimensions the logsumexp
       shall be computed. If -1 (which is also the default), logsumexp is
       computed along the last dimension.
    """
    if len(x.shape) < 2:  # only one possible dimension to sum over?
        xmax = x.max()
        return xmax + np.log(np.sum(np.exp(x - xmax)))
    else:
        if dim != -1:
            x = x.transpose(range(dim) + range(dim + 1, len(x.shape)) + [dim])
        lastdim = len(x.shape) - 1
        xmax = x.max(lastdim)
        return xmax + np.log(np.sum(np.exp(x - xmax[..., None]), lastdim))


def onehot(x, numclasses=None):
    """ Convert integer encoding for class-labels (starting with 0 !)
        to one-hot encoding.

        If numclasses (the number of classes) is not provided, it is assumed
        to be equal to the largest class index occuring in the labels-array+1.
        The output is an array who's shape is the shape of the input array plus
        an extra dimension, containing the 'one-hot'-encoded labels.
    """
    if x.shape == ():
        x = x[None]
    if numclasses is None:
        numclasses = x.max() + 1
    result = np.zeros(list(x.shape) + [numclasses], dtype="int")
    z = np.zeros(x.shape)
    for c in range(numclasses):
        z *= 0
        z[np.where(x == c)] = 1
        result[..., c] += z
    return result


def unhot(labels):
    """ Convert one-hot encoding for class-labels to integer encoding
        (starting with 0!): This can be used to 'undo' a onehot-encoding.

        The input-array can be of any shape. The one-hot encoding is assumed
        to be along the last dimension.
    """
    return labels.argmax(len(labels.shape) - 1)


def gradient_updates_momentum(cost, params, learning_rate, momentum):
    assert momentum < 1 and momentum >= 0
    updates = []
    for param in params:
        param_update = theano.shared(param.get_value() * 0.,
                                     broadcastable=param.broadcastable)
        updates.append((param, param - learning_rate * param_update))
        updates.append((param_update, momentum * param_update +
                        (1. - momentum) * T.grad(cost, param)))
    return updates
