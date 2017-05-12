from __future__ import print_function

import datetime
import os
import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

from utils import get_optimizer, get_loss


class Trainer:
    logdir = None
    session = None
    dataset = None
    model = None
    saver = None
    learning_rate = None
    train_pos_ratio = None
    test_pos_ratio = None
    ckpt_time = None

    def __init__(self, model=None, train_gen=None, test_gen=None, valid_gen=None,
                 optimizer='adam', loss='weighted', pos_weight=1.0,
                 n_epoch=1, train_per_epoch=100000, test_per_epoch=10000,
                 batch_size=2000, learning_rate=1e-2, decay_rate=0.95,
                 logdir=None, load_ckpt=False, ckpt_time=10,
                 layer_keeps=None, percentile=False):
        self.model = model
        self.train_gen = train_gen
        self.test_gen = test_gen
        self.valid_gen = valid_gen
        optimizer = get_optimizer(optimizer)
        loss = get_loss(loss)
        self.pos_weight = pos_weight
        self.n_epoch = n_epoch
        self.train_per_epoch = train_per_epoch
        self.test_per_epoch = test_per_epoch
        self.batch_size = batch_size
        self._learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.logdir = logdir
        self.ckpt_time = ckpt_time
        self.layer_keeps = layer_keeps
        self.percentile = percentile

        self.auc_calculator = roc_auc_score
        config = tf.ConfigProto(allow_soft_placement=True,
                                log_device_placement=False,
                                # device_count={'GPU': 0},
                                )
        config.gpu_options.allow_growth = True
        # config.log_device_placement=True
        self.session = tf.Session(config=config)

        self.learning_rate = tf.placeholder("float")
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        tf.summary.scalar('global_step', self.global_step)
        self.model.compile(loss=loss, optimizer=optimizer(learning_rate=self.learning_rate),
                           global_step=self.global_step, pos_weight=pos_weight)
        self.saver = tf.train.Saver(max_to_keep=1000)
        if train_gen is not None:
            self.train_writer = tf.summary.FileWriter(logdir=os.path.join(logdir, 'train'), graph=self.session.graph,
                                                      flush_secs=30)
        if test_gen is not None:
            self.test_writer = tf.summary.FileWriter(logdir=os.path.join(logdir, 'test'), graph=self.session.graph,
                                                     flush_secs=30) #---------------test ->val
        if valid_gen is not None:
            self.val_writer = tf.summary.FileWriter(logdir=os.path.join(logdir, 'val'), graph=self.session.graph,
                                                    flush_secs=30)
        # if percentile:
        #     self.percentile_write_15 = tf.summary.FileWriter(logdir=os.path.join(logdir, 'percentile_15'),
        #                                                      graph=self.session.graph, flush_secs=30)
        #     self.percentile_write_50 = tf.summary.FileWriter(logdir=os.path.join(logdir, 'percentile_50'),
        #                                                      graph=self.session.graph, flush_secs=30)
        #     self.percentile_write_85 = tf.summary.FileWriter(logdir=os.path.join(logdir, 'percentile_85'),
        #                                                      graph=self.session.graph, flush_secs=30)

        self.session.run(tf.global_variables_initializer())
        self.session.run(tf.local_variables_initializer())
        if load_ckpt:
            ckpt = tf.train.get_checkpoint_state(os.path.join(logdir, 'checkpoints'))
            if ckpt and ckpt.model_checkpoint_path:
                self.saver.restore(self.session, ckpt.model_checkpoint_path)
                print('loaded pre-trained model:', ckpt.model_checkpoint_path)
            else:
                print('No pre-trained model loaded')

    def _run(self, fetches, feed_dict):
        return self.session.run(fetches=fetches, feed_dict=feed_dict)

    def _train(self, X, y):
        feed_dict = {
            self.model.labels: y,
            self.learning_rate: self._learning_rate
        }
        if type(self.model.inputs) is list:
            for i in range(len(self.model.inputs)):
                feed_dict[self.model.inputs[i]] = X[i]
        else:
            feed_dict[self.model.inputs] = X
        if hasattr(self.model, 'layer_keeps'):
            feed_dict[self.model.layer_keeps] = self.layer_keeps
        if hasattr(self.model, 'training'):
            feed_dict[self.model.training] = True
        return self._run(fetches=[self.model.optimizer, self.model.loss, self.model.outputs], feed_dict=feed_dict)

    def _watch(self, X, y, training, watch_list):
        feed_dict = {
            self.model.labels: y,
            self.learning_rate: self._learning_rate
        }
        if type(self.model.inputs) is list:
            for i in range(len(self.model.inputs)):
                feed_dict[self.model.inputs[i]] = X[i]
        else:
            feed_dict[self.model.inputs] = X
        if hasattr(self.model, 'layer_keeps'):
            feed_dict[self.model.layer_keeps] = self.layer_keeps
        if hasattr(self.model, 'training'):
            feed_dict[self.model.training] = training
        fetches = [self.model.optimizer, self.model.loss]
        fetches.extend(watch_list)
        return self._run(fetches=fetches, feed_dict=feed_dict)

    def _predict(self, X, y):
        feed_dict = {
            self.model.labels: y
        }
        if type(self.model.inputs) is list:
            for i in range(len(self.model.inputs)):
                feed_dict[self.model.inputs[i]] = X[i]
        else:
            feed_dict[self.model.inputs] = X
        if hasattr(self.model, 'layer_keeps'):
            feed_dict[self.model.layer_keeps] = np.ones_like(self.layer_keeps)
        if hasattr(self.model, 'training'):
            feed_dict[self.model.training] = False
        return self._run(fetches=[self.model.loss, self.model.outputs], feed_dict=feed_dict)

    def fit(self):
        tic = time.time()
        start_time = time.time()
        total_batches = (self.n_epoch * (self.train_per_epoch + self.test_per_epoch) / self.batch_size)
        finished_batches = 0
        for epoch in range(1, self.n_epoch + 1):
            number_of_batches = ((self.train_per_epoch + self.batch_size - 1) / self.batch_size)
            print('number_of_batches = ', number_of_batches)
            avg_loss = 0
            label_list = []
            pred_list = []
            tx = []
            for batch in range(1, number_of_batches + 1):
                t1 = time.time()
                toc = time.time()
                if toc - tic > self.ckpt_time * 60:
                    print('saving checkpoint...')
                    if not os.path.exists(os.path.join(self.logdir, 'checkpoints')):
                        os.makedirs(os.path.join(self.logdir, 'checkpoints'))
                    self.saver.save(self.session, os.path.join(self.logdir, 'checkpoints', 'model.ckpt'),
                                    self.global_step.eval(self.session))
                    tic = toc
                X, y = self.train_gen.next()
                y = y.squeeze()
                label_list.extend(y)
                t2 = time.time()
                if not self.percentile:
                    _, batch_loss, batch_pred = self._train(X, y)
                    pred_list.extend(batch_pred.flatten())
                else:
                    # watch_list = [self.model.embed_activations, self.model.linear_activations,
                    #               self.model.product_activations, self.model.mixed_activations]
                    # _, batch_loss, embed_act, lin_act, prod_act, mix_act = self._watch(X, y, True, watch_list)
                    # for i, _act in enumerate([embed_act, lin_act, prod_act, mix_act]):
                    #     if i == 0:
                    #         prefix = 'embed'
                    #     elif i == 1:
                    #         prefix = 'linear'
                    #     elif i == 2:
                    #         prefix = 'product'
                    #     elif i == 3:
                    #         prefix = 'mixed'
                    #     _percentile_15 = np.percentile(_act, q=15, axis=0)
                    #     _percentile_50 = np.percentile(_act, q=50, axis=0)
                    #     _percentile_85 = np.percentile(_act, q=85, axis=0)
                    #     for j in range(len(_percentile_15)):
                    #         summary = tf.Summary(value=[tf.Summary.Value(tag='%s_node%d_percentile' % (prefix, j),
                    #                                                      simple_value=_percentile_15[j])])
                    #         self.percentile_write_15.add_summary(summary,
                    #                                              global_step=self.global_step.eval(self.session))
                    #         summary = tf.Summary(value=[tf.Summary.Value(tag='%s_node%d_percentile' % (prefix, j),
                    #                                                      simple_value=_percentile_50[j])])
                    #         self.percentile_write_50.add_summary(summary,
                    #                                              global_step=self.global_step.eval(self.session))
                    #         summary = tf.Summary(value=[tf.Summary.Value(tag='%s_node%d_percentile' % (prefix, j),
                    #                                                      simple_value=_percentile_85[j])])
                    #         self.percentile_write_85.add_summary(summary,
                    #                                              global_step=self.global_step.eval(self.session))
                    pass
                t3 = time.time()
                avg_loss += batch_loss
                finished_batches += 1
                summary = tf.Summary(value=[tf.Summary.Value(tag='loss', simple_value=batch_loss), ])
                self.train_writer.add_summary(summary, global_step=self.global_step.eval(self.session))
                t4 = time.time()
                tx.append([t2-t1, t3-t2, t4-t3])
                if batch % 100 == 0:
                    print(np.mean(tx, axis=0), np.sum(tx, axis=0))
                    tx = []
                    avg_loss /= 100
                    avg_auc = self.auc_calculator(y_true=label_list, y_score=pred_list)
                    elapsed = int(time.time() - start_time)
                    eta = int(1.0 * (total_batches - finished_batches) / finished_batches * elapsed)
                    summary = tf.Summary(value=[tf.Summary.Value(tag='auc', simple_value=avg_auc), ])
                    self.train_writer.add_summary(summary, global_step=self.global_step.eval(self.session))
                    label_list = []
                    pred_list = []
                    print("elapsed : %s, ETA : %s" %
                          (str(datetime.timedelta(seconds=elapsed)), str(datetime.timedelta(seconds=eta))))
                    print('epoch %d / %d, batch %d / %d, global_step = %d, learning_rate = %e, loss = %f, auc = %f'
                          % (epoch, self.n_epoch, batch, number_of_batches, self.global_step.eval(self.session),
                             self._learning_rate, avg_loss, avg_auc))
                    avg_loss = 0
            self._learning_rate *= self.decay_rate
            number_of_batches = ((self.test_per_epoch + self.batch_size - 1) / self.batch_size)
            preds = []
            labels = []
            loss = []

            print('running test...')
            for batch in range(1, number_of_batches + 1):
                X, y = self.test_gen.next() 
                y = np.squeeze(y)
                batch_loss, batch_pred = self._predict(X, y)
                finished_batches += 1
                loss.append(batch_loss)
                preds.append(batch_pred)
                labels.append(y)
            preds = np.concatenate(preds)
            labels = np.concatenate(labels)
            loss = np.mean(loss)
            auc = self.auc_calculator(y_score=preds, y_true=labels)
            if auc < 1e-5:
                auc = self.auc_calculator(y_score=preds, y_true=labels)
            summary = tf.Summary(value=[
                tf.Summary.Value(tag='loss', simple_value=loss),
                tf.Summary.Value(tag='auc', simple_value=auc)
            ])
            self.test_writer.add_summary(summary, global_step=self.global_step.eval(self.session))
            print('test loss = %f, test auc = %f' % (loss, auc))
            print('saving checkpoint...')
            if not os.path.exists(os.path.join(self.logdir, 'checkpoints')):
                os.makedirs(os.path.join(self.logdir, 'checkpoints'))
            self.saver.save(self.session, os.path.join(self.logdir, 'checkpoints', 'model.ckpt'),
                            self.global_step.eval(self.session))
