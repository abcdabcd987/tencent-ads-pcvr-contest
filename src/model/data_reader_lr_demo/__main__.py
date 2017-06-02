import os
import sys
import argparse
import zipfile
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from pprint import pprint
from datetime import datetime

from ...data import *
from ...utils import *
config = read_module_config(__file__, 'model.json')


class LogisticRegressionCTR(object):
    def __init__(self, data_storage, **kwargs):
        self._rep = data_storage.get_representation(IndexRepresentation)
        self._num_feature = self._rep.dense_shape
        self._num_one = self._rep.max_length
        self._learning_rate = config['learning_rate']
        self._batch_size = config['batch_size']

        self._build()

    def _build(self):
        self._ph_x = tf.placeholder(tf.int32, [None, self._num_one])
        self._ph_y = tf.placeholder(tf.float32, [None])
        w = tf.get_variable('weight', [self._num_feature], dtype=tf.float32,
                            initializer=tf.random_uniform_initializer(-0.05, 0.05))
        b = tf.get_variable('bias', [1], dtype=tf.float32,
                            initializer=tf.zeros_initializer())
        wx = tf.reduce_sum(tf.gather(w, self._ph_x), axis=1)
        logits = wx + b
        self._prob = tf.sigmoid(logits)
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self._ph_y, logits=logits))
        opt = tf.train.AdamOptimizer(self._learning_rate)
        self._train_step = opt.minimize(loss)
        self._auc, self._auc_op = tf.metrics.auc(self._ph_y, self._prob)
        tf.summary.scalar('train_loss', loss)
        tf.summary.scalar('train_auc', self._auc)
        self._summary_step = tf.summary.merge_all()

        self._model_dir = os.path.join(config['models_dir'], datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_' + config['module_name'])
        log_dir = os.path.join(config['logs_dir'], datetime.now().strftime('%Y-%m-%d_%H-%M-%S' + '_' + config['module_name']))
        os.makedirs(self._model_dir)
        os.makedirs(log_dir)
        self._saver = tf.train.Saver()
        cfg = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        cfg.gpu_options.allow_growth = True
        self._sess = tf.Session(config=cfg)
        self._summary_writer = tf.summary.FileWriter(log_dir, self._sess.graph, flush_secs=2)

        self._sess.run(tf.global_variables_initializer())
        self._step = 0

    def load(self, model_path):
        checkpoint = tf.train.get_checkpoint_state(model_path)
        if checkpoint and checkpoint.model_checkpoint_path:
            self._saver.restore(self._sess, checkpoint.model_checkpoint_path)
            print("model loaded:", checkpoint.model_checkpoint_path)
        else:
            raise Exception("no model found in " + model_path)

    def save(self):
        filename = os.path.join(self._model_dir, 'model')
        self._saver.save(self._sess, filename, global_step=self._step)

    def train(self, dataset):
        print('training on', dataset)
        self._sess.run(tf.local_variables_initializer())
        for xs, ys, rowids in tqdm(self._rep.get_dataset(dataset, self._batch_size)):
            self._step += 1
            _, _, summary = self._sess.run([self._train_step, self._auc_op, self._summary_step],
                                            feed_dict={self._ph_x: xs, self._ph_y: ys})
            if self._step % 16 == 0:
                self._sess.run(tf.local_variables_initializer())
                self._summary_writer.add_summary(summary, self._step)
                self._summary_writer.flush()

    def validate(self, dataset):
        print('validating on', dataset)
        self._sess.run(tf.local_variables_initializer())
        for xs, ys, rowids in tqdm(self._rep.get_dataset(dataset, self._batch_size)):
            self._sess.run(self._auc_op, feed_dict={self._ph_x: xs, self._ph_y: ys})
        auc = self._sess.run(self._auc)
        summary = tf.Summary(value=[tf.Summary.Value(tag='val_auc', simple_value=auc)])
        self._summary_writer.add_summary(summary, self._step)

    def test(self, dataset):
        print('testing on', dataset)
        res = []
        for xs, ys, rowids in tqdm(self._rep.get_dataset(dataset, self._batch_size)):
            probs = self._sess.run(self._prob, feed_dict={self._ph_x: xs})
            for i, prob in zip(ys, probs):
                res.append((i, prob))
        res.sort(key=lambda i_prob: i_prob[0])

        test_dir = config['tests_dir']
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        filename = os.path.join(test_dir, datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_' + config['module_name'])
        with open(filename + '.csv', 'w') as f:
            f.write('instanceID,prob\n')
            for i, prob in res:
                f.write('{:d},{:.16f}\n'.format(i, prob))
        with zipfile.ZipFile(filename + '.zip', 'w', zipfile.ZIP_DEFLATED) as z:
            z.write(filename + '.csv', 'submission.csv')
        print('test result wrote to', filename + '.csv')
    
    def write_results(self):
        datasets = ['train', 'test']
        res = []
        for dataset in datasets:
            print('predicting on', dataset)
            for xs, ys, rowids in tqdm(self._rep.get_dataset(dataset, self._batch_size)):
                probs = self._sess.run(self._prob, feed_dict={self._ph_x: xs})
                for i, prob in zip(rowids, probs):
                    res.append((i, prob))
        res.sort(key=lambda i_prob: i_prob[0])
        dirname = os.path.join(config['results_dir'], config['module_name'])
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        filename = os.path.join(dirname, datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.txt')
        with open(filename, 'w') as f:
            for i, prob in res:
                f.write('{:.16e}\n'.format(prob))


def main():
    data_storage = DataStorage(config['features'])
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--model', type=str, help='load model from the given path')
    # todo: how about loading model? should it be in the config file?
    args, _ = parser.parse_known_args()

    model = LogisticRegressionCTR(data_storage)
    data_storage.load_data()
    print('feature data loaded')
    if args.model:
        model.load(args.model)
    if args.train:
        print('training...')
        for i in range(4):
            for _ in range(config['epoch']):
                model.train('smalltrain{}'.format(i + 1))
                model.save()
                model.validate('val{}'.format(i + 1))
    if args.test:
        print('testing...')
        model.test('test')
    if args.predict:
        print('predicting for ensemble model...')
        model.write_results()


main()
