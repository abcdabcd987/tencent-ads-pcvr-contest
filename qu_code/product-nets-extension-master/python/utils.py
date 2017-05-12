import numpy as np
import tensorflow as tf

import __init__

minval = __init__.config['minval']
maxval = __init__.config['maxval']
mean = __init__.config['mean']
stddev = __init__.config['stddev']


def get_variable(init_type='tnormal', shape=None, name=None, act_func=None, minval=minval, maxval=maxval, mean=mean,
                 stddev=stddev, dtype=tf.float32, ):
    if init_type == 'tnormal':
        if mean is None:
            mean = 0.0
        if stddev is None:
            if act_func == 'relu':
                stddev = np.sqrt(2 / np.prod(shape))
            else:
                stddev = np.sqrt(1 / np.prod(shape))
        return tf.Variable(tf.truncated_normal(shape=shape, mean=mean, stddev=stddev, dtype=dtype), name=name)
    elif init_type == 'uniform':
        if maxval is None:
            if act_func == 'relu':
                maxval = np.sqrt(2 / np.prod(shape))
            else:
                maxval = np.sqrt(1 / np.prod(shape))
        if minval is None:
            minval = -maxval
        return tf.Variable(tf.random_uniform(shape=shape, minval=minval, maxval=maxval, dtype=dtype), name=name)
    elif init_type == 'normal':
        if mean is None:
            mean = 0.0
        if stddev is None:
            if act_func == 'relu':
                stddev = np.sqrt(2 / np.prod(shape))
            else:
                stddev = np.sqrt(1 / np.prod(shape))
            return tf.Variable(tf.random_normal(shape=shape, mean=mean, stddev=stddev, dtype=dtype), name=name)
    elif init_type == 'zero':
        return tf.Variable(tf.zeros(shape=shape, dtype=dtype), name=name)
    elif init_type == 'one':
        return tf.Variable(tf.ones(shape=shape, dtype=dtype), name=name)
    elif init_type == 'identity' and len(shape) == 2 and shape[0] == shape[1]:
        return tf.Variable(tf.diag(tf.ones(shape=shape[0], dtype=dtype)), name=name)
    elif 'int' in init_type.__class__.__name__ or 'float' in init_type.__class__.__name__:
        return tf.Variable(tf.ones(shape=shape, dtype=dtype) * init_type, name=name)


def activate(weights, activation_function):
    if activation_function == 'sigmoid':
        return tf.nn.sigmoid(weights)
    elif activation_function == 'softmax':
        return tf.nn.softmax(weights)
    elif activation_function == 'relu':
        return tf.nn.relu(weights)
    elif activation_function == 'tanh':
        return tf.nn.tanh(weights)
    elif activation_function == 'elu':
        return tf.nn.elu(weights)
    elif activation_function == 'none':
        return weights
    else:
        return weights


def get_optimizer(opt_algo):
    opt_algo = opt_algo.lower()
    if opt_algo == 'adaldeta':
        return tf.train.AdadeltaOptimizer
    elif opt_algo == 'adagrad':
        return tf.train.AdagradOptimizer
    elif opt_algo == 'adam':
        return tf.train.AdamOptimizer
    elif opt_algo == 'ftrl':
        return tf.train.FtrlOptimizer
    elif opt_algo == 'gd' or opt_algo == 'sgd':
        return tf.train.GradientDescentOptimizer
    elif opt_algo == 'padagrad':
        return tf.train.ProximalAdagradOptimizer
    elif opt_algo == 'pgd':
        return tf.train.ProximalGradientDescentOptimizer
    elif opt_algo == 'rmsprop':
        return tf.train.RMSPropOptimizer
    else:
        return tf.train.GradientDescentOptimizer


def get_loss(loss_func):
    loss_func = loss_func.lower()
    if loss_func == 'weight' or loss_func == 'weighted':
        return tf.nn.weighted_cross_entropy_with_logits
    elif loss_func == 'sigmoid':
        return tf.nn.sigmoid_cross_entropy_with_logits
    elif loss_func == 'softmax':
        return tf.nn.softmax_cross_entropy_with_logits


def normalize(norm, x, n):
    if norm:
        return x / np.sqrt(n)
    else:
        return x


def mul_noise(noisy, x, training):
    if noisy > 0:
        noise = tf.truncated_normal(
            shape=tf.shape(x),
            mean=1.0, stddev=noisy)
        return tf.where(
            training,
            tf.multiply(x, noise),
            x)
    else:
        return x


def add_noise(noisy, x, training):
    if noisy > 0:
        noise = tf.truncated_normal(
            shape=tf.shape(x),
            mean=0, stddev=noisy)
        return tf.where(
            training,
            x + noise,
            x)
    else:
        return x


def gather_2d(params, indices):
    shape = tf.shape(params)
    flat = tf.reshape(params, [-1])
    flat_idx = indices[:, 0] * shape[1] + indices[:, 1]
    flat_idx = tf.reshape(flat_idx, [-1])
    return tf.gather(flat, flat_idx)


def gather_3d(params, indices):
    shape = tf.shape(params)
    flat = tf.reshape(params, [-1])
    flat_idx = indices[:, 0] * shape[1] * shape[2] + indices[:, 1] * shape[2] + indices[:, 2]
    flat_idx = tf.reshape(flat_idx, [-1])
    return tf.gather(flat, flat_idx)


def gather_4d(params, indices):
    shape = tf.shape(params)
    flat = tf.reshape(params, [-1])
    flat_idx = indices[:, 0] * shape[1] * shape[2] * shape[3] + \
               indices[:, 1] * shape[2] * shape[3] + indices[:, 2] * shape[3] + indices[:, 3]
    flat_idx = tf.reshape(flat_idx, [-1])
    return tf.gather(flat, flat_idx)


def max_pool_2d(params, k):
    _, indices = tf.nn.top_k(params, k, sorted=False)
    shape = tf.shape(indices)
    r1 = tf.reshape(tf.range(shape[0]), [-1, 1])
    r1 = tf.tile(r1, [1, k])
    r1 = tf.reshape(r1, [-1, 1])
    indices = tf.concat([r1, tf.reshape(indices, [-1, 1])], 1)
    return tf.reshape(gather_2d(params, indices), [-1, k])


def max_pool_3d(params, k):
    _, indices = tf.nn.top_k(params, k, sorted=False)
    shape = tf.shape(indices)
    r1 = tf.reshape(tf.range(shape[0]), [-1, 1])
    r2 = tf.reshape(tf.range(shape[1]), [-1, 1])
    r1 = tf.tile(r1, [1, k * shape[1]])
    r2 = tf.tile(r2, [1, k])
    r1 = tf.reshape(r1, [-1, 1])
    r2 = tf.tile(tf.reshape(r2, [-1, 1]), [shape[0], 1])
    indices = tf.concat([r1, r2, tf.reshape(indices, [-1, 1])], 1)
    return tf.reshape(gather_3d(params, indices), [-1, shape[1], k])


def max_pool_4d(params, k):
    _, indices = tf.nn.top_k(params, k, sorted=False)
    shape = tf.shape(indices)
    r1 = tf.reshape(tf.range(shape[0]), [-1, 1])
    r2 = tf.reshape(tf.range(shape[1]), [-1, 1])
    r3 = tf.reshape(tf.range(shape[2]), [-1, 1])
    r1 = tf.tile(r1, [1, shape[1] * shape[2] * k])
    r2 = tf.tile(r2, [1, shape[2] * k])
    r3 = tf.tile(r3, [1, k])
    r1 = tf.reshape(r1, [-1, 1])
    r2 = tf.tile(tf.reshape(r2, [-1, 1]), [shape[0], 1])
    r3 = tf.tile(tf.reshape(r3, [-1, 1]), [shape[0] * shape[1], 1])
    indices = tf.concat([r1, r2, r3, tf.reshape(indices, [-1, 1])], 1)
    return tf.reshape(gather_4d(params, indices), [-1, shape[1], shape[2], k])
