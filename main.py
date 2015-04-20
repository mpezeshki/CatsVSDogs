# Credits to Kyle Kastner
from load_data import load_color
import numpy as np
import theano
import theano.tensor as T
from convnet import (conv_layer, fc_layer,
                     softmax_layer, cross_entropy,
                     minibatch_indices)
from utils import gradient_updates_momentum


print "Building model ..."
X = T.tensor4('X')
y = T.ivector('y')
params = []
rng = np.random.RandomState(1999)

# n_filters, n_channels, kernel_width, kernel_height
filter_shape = (32, 3, 3, 3)
pool_shape = (2, 2)
stride = (2, 2)
out, l_params = conv_layer(X, filter_shape, pool_shape, stride, rng)
params += l_params

filter_shape = (64, filter_shape[0], 2, 2)
pool_shape = (2, 2)
stride = (1, 1)
out, l_params = conv_layer(out, filter_shape, pool_shape, stride, rng)
params += l_params

filter_shape = (128, filter_shape[0], 2, 2)
pool_shape = (2, 2)
stride = (1, 1)
out, l_params = conv_layer(out, filter_shape, pool_shape, stride, rng)
params += l_params

shp = out.shape
out = out.reshape((shp[0], shp[1] * shp[2] * shp[3]))  # flatten

shape = (512, 128)
out, l_params = fc_layer(out, shape, rng)
params += l_params

shape = (shape[1], 64)
out, l_params = fc_layer(out, shape, rng)
params += l_params

shape = (shape[1], 2)
out, l_params = softmax_layer(out, shape, rng)
params += l_params

cost = cross_entropy(out, y)
# NaNs
# cost = cost + 0.1 * T.sum([T.sum(grad_i ** 2) for grad_i in grads])
# grads = T.grad(cost, params)
print "Building done.\n"

print "Loading data ..."
X_t, y_t = load_color()
shp = X_t.shape
print shp
# -1 in reshape denotes unknown dimension
X_train = X_t[:20000].reshape(20000, -1).astype('uint8')
mean = X_train.mean(axis=0, keepdims=True)
std = X_train.std(axis=0, keepdims=True)
X_t = (X_t[:].reshape(len(X_t), -1) - mean) / std
X_t = X_t.reshape(*shp)
print "Data loaded.\n"

print "Training model ..."
minibatch_size = 10
learning_rate = 0.01 / minibatch_size
momentum = 0.9

updates = gradient_updates_momentum(cost, params, learning_rate, momentum)
# updates = [(param_i, param_i - learning_rate * grad_i)
#           for param_i, grad_i in zip(params, grads)]

train_function = theano.function([X, y], cost, updates=updates)
predict_function = theano.function([X], out)
epochs = 100
for n in range(epochs):
    loss = []
    for i, j in minibatch_indices(X_t, minibatch_size, lb=0, ub=20000):
        X_nt = X_t[i:j]
        # Random horizontal flips with probability 0.5
        flip_idx = np.where(rng.rand(len(X_nt)) > 0.5)[0]
        X_nt[flip_idx] = X_nt[flip_idx][:, :, :, ::-1]
        l = train_function(X_nt, y_t[i:j])
        loss.append(l)
    loss = np.mean(loss)
    train_y_pred = []
    for i, j in minibatch_indices(X_t, minibatch_size, lb=0, ub=20000):
        train_y_hat = predict_function(X_t[i:j])
        y_p = np.argmax(train_y_hat, axis=1)
        train_y_pred.extend(list(y_p))
    valid_y_pred = []
    for i, j in minibatch_indices(X_t, minibatch_size, lb=20000, ub=25000):
        valid_y_hat = predict_function(X_t[i:j])
        y_p = np.argmax(valid_y_hat, axis=1)
        valid_y_pred.extend(list(y_p))
    train_y_pred = np.array(train_y_pred)
    valid_y_pred = np.array(valid_y_pred)

    print("Epoch %i" % n)
    print("Train Accuracy % f" % np.mean((y_t[0:20000].flatten() ==
                                          train_y_pred.flatten()).astype(
                                              "float32")))
    print("Valid Accuracy % f" % np.mean((y_t[20000:25000].flatten() ==
                                          valid_y_pred.flatten()).astype(
                                              "float32")))
    print("Loss %f" % loss)
