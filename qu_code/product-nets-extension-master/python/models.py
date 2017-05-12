from __future__ import print_function

from abc import abstractmethod

import tensorflow as tf
import numpy as np
import __init__
import utils

dtype = __init__.config['dtype']

if dtype.lower() == 'float32' or dtype.lower() == 'float':
    dtype = tf.float32
elif dtype.lower() == 'float64':
    dtype = tf.float64


def as_model(model_name, **kwargs):
    model_name = model_name.lower()
    if model_name == 'lr':
        return LR(**kwargs)
    elif model_name == 'fm':
        return FM(**kwargs)
    elif model_name == 'ffm':
        return FFM(**kwargs)
    elif model_name == 'rffm':
        return rFFM(**kwargs)
    elif model_name == 'netffm':
        return NetFFM(**kwargs)
    elif model_name == 'fnn':
        return FNN(**kwargs)
    elif model_name == 'pnn1':
        return PNN1(**kwargs)
    elif model_name == 'pnn2':
        return PNN2(**kwargs)


class Model:
    inputs = None
    outputs = None
    logits = None
    labels = None
    learning_rate = None
    loss = None
    optimizer = None

    @abstractmethod
    def compile(self, **kwargs):
        pass

    def __str__(self):
        return self.__class__.__name__


class LR(Model):
    def __init__(self, init='uniform', num_inputs=None, input_dim=None, l2_w=None, norm=False, **kwargs):
        self.l2_w = l2_w
        with tf.name_scope('input'):
            self.inputs = tf.placeholder(tf.int32, [None, num_inputs], name='input')
            self.labels = tf.placeholder(dtype, [None], name='label')

        with tf.name_scope('linear'):
            w = utils.get_variable(init, name='w', shape=[input_dim, 1])
            b = utils.get_variable(0, name='b', shape=[1])
            self.embed = tf.gather(w, self.inputs)
            self.embed = utils.normalize(norm, self.embed, num_inputs)

        with tf.name_scope('output'):
            self.logits = tf.squeeze(tf.reduce_sum(self.embed, 1)) + b
            self.outputs = tf.nn.sigmoid(self.logits)

    def compile(self, loss=None, optimizer=None, global_step=None, pos_weight=1.0, l2=None):
        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(loss(logits=self.logits, targets=self.labels, pos_weight=pos_weight))
            _loss_ = self.loss
            with tf.name_scope('l2_loss'):
                if self.l2_w is not None:
                    _loss_ += self.l2_w * tf.nn.l2_loss(self.embed)
            self.optimizer = optimizer.minimize(loss=_loss_, global_step=global_step)


class FM(Model):
    def __init__(self, init='uniform', num_inputs=None, input_dim=None, factor=10, l2_w=None, l2_v=None, norm=False, **kwargs):
        self.l2_w = l2_w
        self.l2_v = l2_v
        with tf.name_scope('input'):
            self.inputs = tf.placeholder(tf.int32, [None, num_inputs], name='input')
            self.labels = tf.placeholder(dtype, [None], name='label')

        with tf.name_scope('embedding'):
            w = utils.get_variable(init, name='w', shape=[input_dim, 1])
            v = utils.get_variable(init, name='v', shape=[input_dim, factor])
            b = utils.get_variable(0, name='b', shape=[1])

        with tf.name_scope('linear'):
            # batch * fields * 1
            self.xw_embed = tf.gather(w, self.inputs)
            self.xw_embed = utils.normalize(norm, self.xw_embed, num_inputs)
            l = tf.squeeze(tf.reduce_sum(self.xw_embed, 1))

        with tf.name_scope('product'):
            # batch * fields * k
            self.xv_embed = tf.gather(v, self.inputs)
            self.xv_embed = utils.normalize(norm, self.xv_embed, num_inputs)
            # batch * 1
            p = 0.5 * tf.reduce_sum(
                tf.square(
                    tf.reduce_sum(self.xv_embed, 1)) -
                tf.reduce_sum(
                    tf.square(self.xv_embed), 1),
                1)

        with tf.name_scope('output'):
            self.logits = l + b + p
            self.outputs = tf.nn.sigmoid(self.logits)

    def compile(self, loss=None, optimizer=None, global_step=None, pos_weight=1.0):
        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(loss(logits=self.logits, targets=self.labels, pos_weight=pos_weight))
            _loss_ = self.loss
            with tf.name_scope('l2_loss'):
                if self.l2_w is not None:
                    _loss_ += self.l2_w * tf.nn.l2_loss(self.xw_embed)
                if self.l2_v is not None:
                    _loss_ += self.l2_v * tf.nn.l2_loss(self.xv_embed)
            self.optimizer = optimizer.minimize(loss=_loss_, global_step=global_step)


class FFM(Model):
    def __init__(self, init='uniform', num_inputs=None, input_dim=None, factor=10, l2_w=None, l2_v=None, noisy=0,
                 norm=False, **kwargs):
        self.l2_w = l2_w
        self.l2_v = l2_v

        with tf.name_scope('input'):
            self.inputs = tf.placeholder(tf.int32, [None, num_inputs], name='input')
            self.labels = tf.placeholder(dtype, [None], name='label')
            self.training = tf.placeholder(tf.bool, name='training')

        with tf.name_scope('embedding'):
            w = utils.get_variable(init, name='w', shape=[input_dim, 1])
            v = utils.get_variable(init, name='v', shape=[input_dim, num_inputs - 1, factor])
            b = utils.get_variable(0, name='b', shape=[1])

        with tf.name_scope('linear'):
            # batch * fields * 1
            self.xw_embed = tf.gather(w, self.inputs)
            self.xw_embed = utils.normalize(norm, self.xw_embed, num_inputs)
            l = tf.squeeze(tf.reduce_sum(self.xw_embed, 1))

        with tf.name_scope('product'):
            # batch * fields * field * k
            self.xv_embed = tf.gather(v, self.inputs)
            self.xv_embed = utils.normalize(norm, self.xv_embed, num_inputs)
            self.xv_embed = utils.mul_noise(noisy, self.xv_embed, self.training)

            rows = []
            cols = []
            for i in range(num_inputs - 1):
                for j in range(i + 1, num_inputs):
                    rows.append([i, j - 1])
                    cols.append([j, i])
            # batch * pair * factor
            xv_p = tf.transpose(tf.gather_nd(tf.transpose(self.xv_embed, [1, 2, 0, 3]), rows), [1, 0, 2])
            xv_q = tf.transpose(tf.gather_nd(tf.transpose(self.xv_embed, [1, 2, 0, 3]), cols), [1, 0, 2])
            p = tf.reduce_sum(
                tf.reduce_sum(
                    tf.multiply(xv_p, xv_q),
                    2),
                1)

        with tf.name_scope('output'):
            self.logits = l + b + p
            self.outputs = tf.nn.sigmoid(self.logits)

    def compile(self, loss=None, optimizer=None, global_step=None, pos_weight=1.0):
        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(loss(logits=self.logits, targets=self.labels, pos_weight=pos_weight))
            _loss_ = self.loss
            with tf.name_scope('l2_loss'):
                if self.l2_w is not None:
                    _loss_ += self.l2_w * tf.nn.l2_loss(self.xw_embed)
                if self.l2_v is not None:
                    _loss_ += self.l2_v * tf.nn.l2_loss(self.xv_embed)
            self.optimizer = optimizer.minimize(loss=_loss_, global_step=global_step)


class rFFM(Model):
    def __init__(self, init='uniform', num_inputs=None, input_dim=None, factor=10, l2_w=None, l2_v=None, norm=False,
                 noisy=0, **kwargs):
        self.l2_w = l2_w
        self.l2_v = l2_v
        with tf.name_scope('input'):
            self.inputs = tf.placeholder(tf.int32, [None, num_inputs], name='input')
            self.labels = tf.placeholder(dtype, [None], name='label')
            self.training = tf.placeholder(tf.bool, name='training')

        with tf.name_scope('embedding'):
            w = utils.get_variable(init, name='w', shape=[input_dim, 1])
            b = utils.get_variable(0, name='b', shape=[1])
            v = utils.get_variable(init, name='v', shape=[input_dim, factor])

        with tf.name_scope('linear'):
            # batch * fields * 1
            self.xw_embed = tf.gather(w, self.inputs)
            if norm:
                self.xw_embed /= np.sqrt(num_inputs)
            l = tf.squeeze(tf.reduce_sum(self.xw_embed, 1))

        with tf.name_scope('product'):
            # batch * fields * k
            self.xv_embed = tf.gather(v, self.inputs)
            self.xv_embed = utils.normalize(norm, self.xv_embed, num_inputs)
            self.xv_embed = utils.mul_noise(noisy, self.xv_embed, self.training)

            num_pairs = int(num_inputs * (num_inputs - 1) / 2)
            k = utils.get_variable(init, name='k', shape=[factor, num_pairs, factor])
            kb = utils.get_variable(0, name='kb', shape=[num_pairs])
            rows = []
            cols = []
            for i in range(num_inputs - 1):
                for j in range(i + 1, num_inputs):
                    rows.append(i)
                    cols.append(j)
            # batch * 1 * pair * k
            xv_p = tf.expand_dims(
                tf.transpose(
                    tf.gather(
                        tf.transpose(
                            self.xv_embed, [1, 0, 2]),
                        rows),
                    [1, 0, 2]),
                1)
            # batch * pair * k
            xv_q = tf.transpose(
                tf.gather(
                    tf.transpose(
                        self.xv_embed, [1, 0, 2]),
                    cols),
                [1, 0, 2])
            # TODO check tf.tensordot
            prods = tf.reduce_sum(
                tf.multiply(
                    tf.transpose(
                        tf.reduce_sum(
                            tf.multiply(
                                xv_p, k),
                            -1),
                        [0, 2, 1]),
                    xv_q),
                -1) + kb
            p = tf.reduce_sum(
                prods,
                1)

            # k = [[utils.get_variable(init, name='k_%d_%d' % (i, j), shape=[factor, factor])
            #       for j in range(i + 1, num_inputs)]
            #      for i in range(num_inputs - 1)]
            # kb = [[utils.get_variable(0, name='kb_%d_%d' % (i, j), shape=[1])
            #        for j in range(i + 1, num_inputs)]
            #       for i in range(num_inputs - 1)]
            # prods = []
            # cnt = 0
            # for i in range(num_inputs - 1):
            #     for j in range(i + 1, num_inputs):
            #         prods.append(tf.reduce_sum(
            #             tf.multiply(
            #                 tf.matmul(self.xv_embed[:, i, :], k[i][j - i - 1]),
            #                 # tf.matmul(self.xv_embed[:, i, :], k[cnt]),
            #                 self.xv_embed[:, j, :]),
            #             1, keep_dims=True) + kb[i][j - i - 1])
            #             # 1, keep_dims = True) + kb[cnt])
            #         cnt += 1
            # p = tf.reduce_sum(
            #     tf.concat(prods, 1),
            #     1, keep_dims=True)

        with tf.name_scope('output'):
            self.logits = l + p + b
            self.outputs = tf.nn.sigmoid(self.logits)

    def compile(self, loss=None, optimizer=None, global_step=None, pos_weight=1.0):
        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(loss(logits=self.logits, targets=self.labels, pos_weight=pos_weight))
            _loss_ = self.loss
            with tf.name_scope('l2_loss'):
                if self.l2_w is not None:
                    _loss_ += self.l2_w * tf.nn.l2_loss(self.xw_embed)
                if self.l2_v is not None:
                    _loss_ += self.l2_v * tf.nn.l2_loss(self.xv_embed)
            self.optimizer = optimizer.minimize(loss=_loss_, global_step=global_step)


class NetFFM(Model):
    def __init__(self, init='uniform', num_inputs=None, input_dim=None, factor=10, l2_w=None, l2_v=None, norm=False,
                 layer_sizes=None, layer_acts=None, **kwargs):
        self.l2_w = l2_w
        self.l2_v = l2_v

        with tf.name_scope('input'):
            self.inputs = tf.placeholder(tf.int32, [None, num_inputs], name='input')
            self.labels = tf.placeholder(dtype, [None], name='label')

        with tf.name_scope('embedding'):
            w = utils.get_variable(init, name='w', shape=[input_dim, 1])
            v = utils.get_variable(init, name='v', shape=[input_dim, factor])
            b = utils.get_variable(0, name='b', shape=[1])

        with tf.name_scope('linear'):
            # batch * fields * 1
            self.xw_embed = tf.gather(w, self.inputs)
            if norm:
                self.xw_embed /= np.sqrt(num_inputs)
            l = tf.squeeze(tf.reduce_sum(self.xw_embed, 1))

        with tf.name_scope('product'):
            # batch * fields * k
            self.xv_embed = tf.gather(v, self.inputs)
            self.xv_embed = utils.normalize(norm, self.xv_embed, num_inputs)
            prods = []
            for i in range(num_inputs - 1):
                for j in range(i + 1, num_inputs):
                    prods.append(tf.concat([self.xv_embed[:, i, :], self.xv_embed[:, j, :]], 1))
            for i in range(len(layer_sizes) - 1):
                with tf.name_scope('layer_%d' % i):
                    for j in range(len(prods)):
                        _w_ = utils.get_variable(init, name='w_%d_%d' % (i, j),
                                                 shape=[layer_sizes[i], layer_sizes[i + 1]])
                        _b_ = utils.get_variable(0, name='b_%d_%d' % (i, j), shape=[layer_sizes[i + 1]])
                        prods[j] = utils.activate(tf.matmul(prods[j], _w_) + _b_, layer_acts[i])
            p = tf.reduce_sum(
                tf.concat(prods, 1),
                1)

        with tf.name_scope('output'):
            self.logits = l + p + b
            self.outputs = tf.nn.sigmoid(self.logits)

    def compile(self, loss=None, optimizer=None, global_step=None, pos_weight=1.0):
        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(loss(logits=self.logits, targets=self.labels, pos_weight=pos_weight))
            _loss_ = self.loss
            with tf.name_scope('l2'):
                if self.l2_w is not None:
                    _loss_ += self.l2_w * tf.nn.l2_loss(self.xw_embed)
                if self.l2_v is not None:
                    _loss_ += self.l2_v * tf.nn.l2_loss(self.xv_embed)
            self.optimizer = optimizer.minimize(loss=_loss_, global_step=global_step)


class FNN(Model):
    def __init__(self, init='uniform', num_inputs=None, layer_sizes=None, layer_acts=None, layer_l2=None,
                 norm=False, **kwargs):
        input_sizes = layer_sizes[0]
        embed_sizes = layer_sizes[1]
        if type(embed_sizes) is int:
            embed_sizes = [embed_sizes] * num_inputs

        with tf.name_scope('input'):
            self.inputs = [tf.placeholder(tf.int32, [None, 1], name='input_%d' % i) for i in range(num_inputs)]
            self.labels = tf.placeholder(dtype, [None], name='label')
            self.layer_keeps = tf.placeholder(dtype, name='keep_prob')
            self.layer_l2 = layer_l2

        with tf.name_scope('embedding'):
            with tf.name_scope('field'):
                # different size embedding for fields
                w0 = [utils.get_variable(init, name='w0', shape=[input_sizes[i], embed_sizes[i]]) for i in
                      range(num_inputs)]
                # g0 = [utils.get_variable(1, name='g0', shape=[1]) for i in range(num_inputs)]
                # v0 = [w0[i] / tf.nn.l2_loss(w0[i]) * g0[i] for i in range(num_inputs)]
                b0 = [utils.get_variable(0, name='b0', shape=[embed_sizes[i]]) for i in range(num_inputs)]
                embed_0 = [tf.squeeze(tf.gather(w0[i], self.inputs[i])) for i in range(num_inputs)]
                embed_0 = [utils.normalize(norm, x, num_inputs) for x in embed_0]
            x_embed = tf.concat([embed_0[i] + b0[i] for i in range(num_inputs)], 1)
            l = tf.nn.dropout(
                utils.activate(x_embed, layer_acts[0]),
                self.layer_keeps[0])
            self.embed = []
            self.embed.append(tf.concat(embed_0, 1))

        with tf.name_scope('hidden_1'):
            w1 = utils.get_variable(init, name='w1', shape=[sum(embed_sizes), layer_sizes[2]])
            # g1 = utils.get_variable(1, name='g1', shape=[1])
            # v1 = w1 / tf.nn.l2_loss(w1) * g1
            b1 = utils.get_variable(0, name='b1', shape=[layer_sizes[2]])
            l = tf.nn.dropout(
                utils.activate(
                    tf.matmul(l, w1) + b1,
                    layer_acts[1]),
                self.layer_keeps[1])
            self.embed.append(w1)

        for i in range(2, len(layer_sizes) - 1):
            with tf.name_scope('hidden_%d' % i):
                wi = utils.get_variable(init, name='w%d' % i, shape=[layer_sizes[i], layer_sizes[i + 1]])
                # gi = utils.get_variable(1, name='g%d' % i, shape=[1])
                # vi = wi / tf.nn.l2_loss(wi) * gi
                bi = utils.get_variable(0, name='b%d' % i, shape=[layer_sizes[i + 1]])
                l = tf.nn.dropout(
                    utils.activate(
                        tf.matmul(l, wi) + bi,
                        layer_acts[i]),
                    self.layer_keeps[i])
                self.embed.append(wi)

        with tf.name_scope('output'):
            # TODO: !!!!!!!!!!!!!!!!!!!!!!!!check dimension!!!!!!!!!!!!!!!!!!!!!!
            self.logits = tf.squeeze(l)
            self.outputs = tf.nn.sigmoid(self.logits)

    def compile(self, loss=None, optimizer=None, global_step=None, pos_weight=1.0):
        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(loss(logits=self.logits, targets=self.labels, pos_weight=pos_weight))
            _loss_ = self.loss
            with tf.name_scope('l2_loss'):
                if self.layer_l2 is not None:
                    for i in range(len(self.embed)):
                        _loss_ += self.layer_l2[i] * tf.nn.l2_loss(self.embed[i])
            self.optimizer = optimizer.minimize(loss=_loss_, global_step=global_step)


class PNN1(Model):
    def __init__(self, init='uniform', num_inputs=None, layer_sizes=None, layer_acts=None, layer_l2=None,
                 norm=False, **kwargs):
        input_sizes = layer_sizes[0]
        embed_size = layer_sizes[1]

        with tf.name_scope('input'):
            self.inputs = [tf.placeholder(tf.int32, [None, 1], name='input_%d' % i) for i in range(num_inputs)]
            self.labels = tf.placeholder(dtype, [None], name='label')
            self.layer_keeps = tf.placeholder(dtype, name='keep_prob')
            self.training = tf.placeholder(dtype=tf.bool, name='training')
            self.layer_l2 = layer_l2

        with tf.name_scope('embedding'):
            # different size embedding for fields
            w0 = [utils.get_variable(init, name='w0', shape=[input_sizes[i], embed_size]) for i in
                  range(num_inputs)]
            b0 = [utils.get_variable(0, name='b0', shape=[embed_size]) for i in range(num_inputs)]
            embed_0 = [tf.nn.embedding_lookup(w0[i], self.inputs[i]) for i in range(num_inputs)]
            embed_0 = [utils.normalize(norm, x, num_inputs) for x in embed_0]
            x_embed = tf.concat([tf.squeeze(embed_0[i]) + b0[i] for i in range(num_inputs)], 1)
            l = tf.nn.dropout(
                utils.activate(x_embed, layer_acts[0]),
                self.layer_keeps[0])
            self.embed = []
            self.embed.append(tf.concat(embed_0, 1))

        with tf.name_scope('product'):
            w1 = utils.get_variable(init, name='w1', shape=[num_inputs * embed_size, layer_sizes[2]])
            k1 = utils.get_variable(init, name='k1', shape=[num_inputs, layer_sizes[2]])
            b1 = utils.get_variable(0, name='b1', shape=[layer_sizes[2]])
            p = tf.reduce_sum(
                tf.reshape(
                    tf.matmul(
                        tf.reshape(
                            tf.transpose(
                                tf.reshape(l, [-1, num_inputs, embed_size]),
                                [0, 2, 1]),
                            [-1, num_inputs]),
                        k1),
                    [-1, embed_size, layer_sizes[2]]),
                1)
            l = tf.matmul(l, w1) + p
            # if batch_norm:
            #     l = tf.layers.batch_normalization(l, training=self.training)
            l = tf.nn.dropout(
                utils.activate(l + b1, layer_acts[1]),
                self.layer_keeps[1])
            self.embed.append(w1)
            self.embed.append(k1)

        for i in range(2, len(layer_sizes) - 1):
            with tf.name_scope('hidden_%d' % i):
                wi = utils.get_variable(init, name='w%d' % i, shape=[layer_sizes[i], layer_sizes[i + 1]])
                bi = utils.get_variable(0, name='b%d' % i, shape=[layer_sizes[i + 1]])
                l = tf.nn.dropout(
                    utils.activate(
                        tf.matmul(l, wi) + bi,
                        layer_acts[i]),
                    self.layer_keeps[i])
                self.embed.append(wi)

        with tf.name_scope('output'):
            ## TODO: !!!!!!!!!!!!!!!!!!!!!!!!check dimension!!!!!!!!!!!!!!!!!!!!!!
            self.logits = tf.squeeze(l)
            self.outputs = tf.nn.sigmoid(self.logits)

    def compile(self, loss=None, optimizer=None, global_step=None, pos_weight=1.0):
        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(loss(logits=self.logits, targets=self.labels, pos_weight=pos_weight))
            _loss_ = self.loss
            with tf.name_scope('l2_loss'):
                if self.layer_l2 is not None:
                    for i in range(len(self.embed)):
                        _loss_ += self.layer_l2[i] * tf.nn.l2_loss(self.embed[i])
            self.optimizer = optimizer.minimize(loss=_loss_, global_step=global_step)


class PNN2(Model):
    def __init__(self, init='uniform', num_inputs=None, layer_sizes=None, layer_acts=None, layer_l2=None,
                 norm=False, **kwargs):
        input_sizes = layer_sizes[0]
        embed_size = layer_sizes[1]

        with tf.name_scope('input'):
            self.inputs = [tf.placeholder(tf.int32, [None, 1], name='input_%d' % i) for i in range(num_inputs)]
            self.labels = tf.placeholder(dtype, [None], name='label')
            self.layer_keeps = tf.placeholder(dtype, name='keep_prob')
            self.training = tf.placeholder(dtype=tf.bool, name='training')
            self.layer_l2 = layer_l2

        with tf.name_scope('embedding'):
            # different size embedding for fields
            w0 = [utils.get_variable(init, name='w0', shape=[input_sizes[i], embed_size]) for i in
                  range(num_inputs)]
            b0 = [utils.get_variable(0, name='b0', shape=[embed_size]) for i in range(num_inputs)]
            embed_0 = [tf.nn.embedding_lookup(w0[i], self.inputs[i]) for i in range(num_inputs)]
            embed_0 = [utils.normalize(norm, x, num_inputs) for x in embed_0]
            x_embed = tf.concat([tf.squeeze(embed_0[i]) + b0[i] for i in range(num_inputs)], 1)
            l = tf.nn.dropout(
                utils.activate(x_embed, layer_acts[0]),
                self.layer_keeps[0])
            # self.embed_activations = x_embed
            self.embed = []
            self.embed.append(tf.concat(embed_0, 1))

        with tf.name_scope('product'):
            w1 = utils.get_variable(init, name='w1', shape=[num_inputs * embed_size, layer_sizes[2]])
            k1 = utils.get_variable(init, name='k1', shape=[embed_size * embed_size, layer_sizes[2]])
            b1 = utils.get_variable(0, name='b1', shape=[layer_sizes[2]])
            z = tf.reduce_sum(tf.reshape(l, [-1, num_inputs, embed_size]), 1)
            p = tf.reshape(
                tf.matmul(tf.reshape(z, [-1, embed_size, 1]),
                          tf.reshape(z, [-1, 1, embed_size])),
                [-1, embed_size * embed_size])
            # self.linear_activations = tf.matmul(l, w1)
            # self.product_activations = tf.matmul(p, k1)
            # self.mixed_activations = self.linear_activations + self.product_activations + b1
            l = tf.matmul(l, w1) + tf.matmul(p, k1)
            # if batch_norm:
            #     # self.mixed_activations = tf.layers.batch_normalization(self.mixed_activations, training=self.training)
            #     l = tf.layers.batch_normalization(l, training=self.training)
            l = tf.nn.dropout(
                utils.activate(
                    l + b1, layer_acts[1]),
                self.layer_keeps[1])
            self.embed.append(w1)
            self.embed.append(k1)

        for i in range(2, len(layer_sizes) - 1):
            with tf.name_scope('hidden_%d' % i):
                wi = utils.get_variable(init, name='w%d' % i, shape=[layer_sizes[i], layer_sizes[i + 1]])
                bi = utils.get_variable(0, name='b%d' % i, shape=[layer_sizes[i + 1]])
                li_act = tf.matmul(l, wi) + bi
                l = tf.nn.dropout(
                    utils.activate(
                        li_act,
                        layer_acts[i]),
                    self.layer_keeps[i])
                self.embed.append(wi)

        with tf.name_scope('output'):
            ## TODO: !!!!!!!!!!!!!!!!!!!!!!!!check dimension!!!!!!!!!!!!!!!!!!!!!!
            self.logits = tf.squeeze(l)
            self.outputs = tf.nn.sigmoid(self.logits)

    def compile(self, loss=None, optimizer=None, global_step=None, pos_weight=1.0):
        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(loss(logits=self.logits, targets=self.labels, pos_weight=pos_weight))
            _loss_ = self.loss
            with tf.name_scope('l2_loss'):
                if self.layer_l2 is not None:
                    for i in range(len(self.embed)):
                        _loss_ += self.layer_l2[i] * tf.nn.l2_loss(self.embed[i])
            self.optimizer = optimizer.minimize(loss=_loss_, global_step=global_step)
