import os
import abc
import sys
import glob
import gzip
import argparse
import zipfile
import traceback
import multiprocessing
from datetime import datetime
from collections import deque
from pprint import pprint
import numpy as np
import tensorflow as tf


MODEL_NAME = os.path.splitext(os.path.basename(__file__))[0]

class BufferedDataReader(object):
    def __init__(self, filename, batch_size, num_one, batch_buffer_size=100, num_batch_worker=4):
        self._line_queue = multiprocessing.Queue(batch_buffer_size * batch_size)
        self._batch_queue = multiprocessing.Queue(batch_buffer_size)
        self._batch_maker = []
        if batch_size == -1:
            batch_size = sys.maxint
        for _ in xrange(num_batch_worker):
            p = multiprocessing.Process(target=BufferedDataReader._batch_worker,
                                        args=(self._line_queue, self._batch_queue, batch_size, num_one))
            self._batch_maker.append(p)
            p.start()
        self._line_reader = multiprocessing.Process(target=BufferedDataReader._line_worker,
                                                    args=(self._line_queue, filename, len(self._batch_maker)))
        self._line_reader.start()

    def qsize(self):
        return self._line_queue.qsize(), self._batch_queue.qsize()

    def stop(self):
        self._line_reader.terminate()
        for p in self._batch_maker:
            p.terminate()

    def join(self):
        self._line_reader.join()
        for p in self._batch_maker:
            p.join()

    def get_batch(self):
        xs, ys = self._batch_queue.get()
        return xs, ys

    @staticmethod
    def _batch_worker(line_queue, batch_queue, batch_size, num_one):
        run = True
        while run:
            xs = np.zeros((batch_size, num_one), dtype=int)
            ys = np.empty(batch_size, dtype=int)
            for i in xrange(batch_size):
                line = line_queue.get()
                if line is None:
                    xs, ys = xs[:i], ys[:i]
                    run = False
                    break

                split = line.split()
                clicked = int(split[0])
                for j, s in enumerate(split[1:]):
                    k, v = s.split(':')
                    xs[i, j] = int(k)
                ys[i] = clicked
            batch_queue.put((xs, ys))
            run = run and len(ys) != 0
        batch_queue.put(([], []))
        batch_queue.close()

    @staticmethod
    def _line_worker(line_queue, filename, num_batch_maker):
        with gzip.open(filename) as f:
            for line in f:
                line_queue.put(line)
        for _ in xrange(num_batch_maker):
            line_queue.put(None)
        line_queue.close()


class LinearRegressionCTR(object):
    def __init__(self, **kwargs):
        self._data_root = kwargs['data_root']
        self._output_root = kwargs['output_root']
        self._num_feature = kwargs['num_feature']
        self._num_one = kwargs['num_one']
        self._learning_rate = kwargs['learning_rate']
        self._batch_size = kwargs['batch_size']

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

        self._model_dir = os.path.join(self._output_root, 'models', datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_' + MODEL_NAME)
        log_dir = os.path.join(self._output_root, 'logs', datetime.now().strftime('%Y-%m-%d_%H-%M-%S' + '_' + MODEL_NAME))
        os.makedirs(self._model_dir)
        os.makedirs(log_dir)
        self._saver = tf.train.Saver()
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth=True
        self._sess = tf.Session(config=config)
        self._summary_writer = tf.summary.FileWriter(log_dir, self._sess.graph, flush_secs=2)

        self._sess.run(tf.global_variables_initializer())
        self._step = 0
    
    def load(self, model_path):
        checkpoint = tf.train.get_checkpoint_state(model_path)
        if checkpoint and checkpoint.model_checkpoint_path:
            self._saver.restore(self._sess, checkpoint.model_checkpoint_path)
            print "model loaded:", checkpoint.model_checkpoint_path
        else:
            raise Exception("no model found in " + model_path)

    def save(self):
        filename = os.path.join(self._model_dir, 'model')
        self._saver.save(self._sess, filename, global_step=self._step)

    def train(self):
        filename = os.path.join(self._data_root, 'train.txt.gz')
        reader = BufferedDataReader(filename, self._batch_size, self._num_one)
        self._sess.run(tf.local_variables_initializer())
        try:
            while True:
                batch = reader.get_batch()
                xs, ys = batch
                if len(ys) == 0:
                    break
                self._step += 1
                _, _, summary = self._sess.run([self._train_step, self._auc_op, self._summary_step],
                                               feed_dict={self._ph_x: xs, self._ph_y: ys})
                if self._step % 16 == 0:
                    self._sess.run(tf.local_variables_initializer())
                    self._summary_writer.add_summary(summary, self._step)
                    _, num_batch_buffer = reader.qsize()
                    summary = tf.Summary(value=[tf.Summary.Value(tag='batch_buffer', simple_value=num_batch_buffer)])
                    self._summary_writer.add_summary(summary, self._step)
        except:
            reader.stop()
            reader.join()
            raise
        reader.stop()
        reader.join()

    def validate(self):
        filename = os.path.join(self._data_root, 'val.txt.gz')
        reader = BufferedDataReader(filename, self._batch_size, self._num_one)
        self._sess.run(tf.local_variables_initializer())
        try:
            while True:
                xs, ys = reader.get_batch()
                if len(ys) == 0:
                    break
                self._sess.run(self._auc_op, feed_dict={self._ph_x: xs, self._ph_y: ys})
            auc = self._sess.run(self._auc)
            summary = tf.Summary(value=[tf.Summary.Value(tag='val_auc', simple_value=auc)])
            self._summary_writer.add_summary(summary, self._step)
        except:
            reader.stop()
            reader.join()
            raise
        reader.stop()
        reader.join()
    
    def test(self):
        res = []
        filename = os.path.join(self._data_root, 'test.txt.gz')
        reader = BufferedDataReader(filename, self._batch_size, self._num_one)
        try:
            while True:
                xs, ys = reader.get_batch()
                if len(ys) == 0:
                    break
                probs = self._sess.run(self._prob, feed_dict={self._ph_x: xs})
                for i, prob in zip(ys, probs):
                    res.append((i, prob))
        except:
            reader.stop()
            reader.join()
            raise
        reader.stop()
        reader.join()
        res.sort(key=lambda (i, prob): i)

        test_dir = os.path.join(self._output_root, 'tests')
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        filename = os.path.join(test_dir, datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_' + MODEL_NAME)
        with open(filename + '.csv', 'w') as f:
            f.write('instanceID,prob\n')
            for i, prob in res:
                f.write('{:d},{:.16f}\n'.format(i, prob))
        with zipfile.ZipFile(filename + '.zip', 'w', zipfile.ZIP_DEFLATED) as z:
            z.write(filename + '.csv', 'submission.csv')
        print 'test result wrote to', filename + '.csv'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--output_root', type=str, required=True)
    parser.add_argument('--num_feature', type=int, required=True)
    parser.add_argument('--num_one', type=int, required=True)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--model', type=str, help='load model from the given path')
    args, _ = parser.parse_known_args()

    data_root = args.data_root
    output_root = args.output_root
    model = LinearRegressionCTR(data_root=data_root,
                                output_root=output_root,
                                num_feature=args.num_feature,
                                num_one=args.num_one,
                                learning_rate=5e-4,
                                batch_size=512)
    if args.model:
        model.load(args.model)
    if args.train:
        print 'training...'
        for _ in xrange(10):
            model.train()
            model.save()
            model.validate()
    model.test()

if __name__ == '__main__':
    main()