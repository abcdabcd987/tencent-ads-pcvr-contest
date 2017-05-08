import os
import abc
import sys
import glob
import gzip
import argparse
import multiprocessing
from datetime import datetime
from collections import deque
from pprint import pprint
import numpy as np
import tensorflow as tf


MODEL_NAME = os.path.splitext(os.path.basename(__file__))[0]


class LinearRegressionCTR(object):
    def __init__(self, **kwargs):
        self._data_root = kwargs['data_root']
        self._output_root = kwargs['output_root']
        self._num_feature = kwargs['num_feature']
        self._learning_rate = kwargs['learning_rate']
        self._batch_size = kwargs['batch_size']

        self._build()

    def _inputs(self, filenames):
        filename_queue = tf.train.string_input_producer(filenames)
        options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
        reader = tf.TFRecordReader(options=options)
        _, serialized_example = reader.read(filename_queue)
        batch_serialized_examples = tf.train.batch([serialized_example],
                                                   batch_size=self._batch_size,
                                                   num_threads=8,
                                                   capacity=self._batch_size * 128)
        proto = {
            'index': tf.VarLenFeature(tf.int64),
            'value': tf.VarLenFeature(tf.float32),
            'label': tf.FixedLenFeature(shape=[1], dtype=tf.int64),
        }
        features = tf.parse_example(batch_serialized_examples, proto)
        indicies = features['index']
        values = features['value']
        labels = tf.cast(tf.squeeze(features['label'], axis=1), tf.float32)
        return indicies, values, labels

    def _build(self):
        filename = os.path.join(self._data_root, 'train.tfrecord.gz')
        indicies, values, labels = self._inputs([filename])

        # self._ph_x = tf.placeholder(tf.int32, [None, None])
        # self._ph_y = tf.placeholder(tf.float32, [None])
        w = tf.get_variable('weight', [self._num_feature], dtype=tf.float32,
                            initializer=tf.random_uniform_initializer(-0.05, 0.05))
        b = tf.get_variable('bias', [1], dtype=tf.float32,
                            initializer=tf.zeros_initializer())
        wx = tf.nn.embedding_lookup_sparse(params=w, sp_ids=indicies, sp_weights=values, combiner='sum')
        # wx = tf.sparse_tensor_dense_matmul(self._ph_x, w)
        # wx = tf.reduce_sum(tf.gather(w, self._ph_x), axis=1)
        logits = wx + b
        prob = tf.sigmoid(logits)
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))
        # loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=labels, logits=logits, pos_weight=100))
        opt = tf.train.AdamOptimizer(self._learning_rate)
        self._train_step = opt.minimize(loss)
        self._auc, self._auc_op = tf.metrics.auc(labels, prob)
        tf.summary.scalar('train_loss', loss)
        tf.summary.scalar('train_auc', self._auc)
        self._summary_step = tf.summary.merge_all()

        self._model_dir = os.path.join(self._output_root, 'models', datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_' + MODEL_NAME)
        log_dir = os.path.join(self._output_root, 'logs', datetime.now().strftime('%Y-%m-%d_%H-%M-%S' + '_' + MODEL_NAME))
        if self._output_root.startswith('gs://'):
            # create directory for model saver path
            bucket, path = get_bucket_and_path(os.path.split(self._model_dir)[0])
            blob = bucket.blob(path)
            blob.upload_from_string('')
        else:
            os.makedirs(self._model_dir)
            os.makedirs(log_dir)
        self._saver = tf.train.Saver()
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth=True
        self._sess = tf.Session(config=config)
        self._summary_writer = tf.summary.FileWriter(log_dir, self._sess.graph, flush_secs=2)

        self._sess.run(tf.global_variables_initializer())
        self._step = 0

    def save(self):
        self._saver.save(self._sess, self._model_dir, global_step=self._step)

    def train(self):
        filename = os.path.join(self._data_root, 'train.tfrecord.gz')
        # xs, ys = self._inputs([filename])
        self._sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self._sess, coord=coord)
        try:
            while not coord.should_stop():
                self._step += 1
                _, _, summary = self._sess.run([self._train_step, self._auc_op, self._summary_step])
                # _, _, summary = self._sess.run([self._train_step, self._auc_op, self._summary_step],
                #                                feed_dict={self._ph_x: xs, self._ph_y: ys})
                if self._step % 16 == 0:
                    self._sess.run(tf.local_variables_initializer())
                    self._summary_writer.add_summary(summary, self._step)
        except tf.errors.OutOfRangeError:
            pass
        finally:
            coord.request_stop()
        coord.join(threads)

    def validate(self):
        filename = os.path.join(self._data_root, 'val.tfrecord.gz')
        xs, ys = self._inputs(filename, self._batch_size)
        self._sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(self._sess, coord=coord)
        try:
            while not coord.should_stop():
                self._sess.run(self._auc_op, feed_dict={self._ph_x: xs, self._ph_y: ys})
            auc = self._sess.run(self._auc)
            summary = tf.Summary(value=[tf.Summary.Value(tag='val_auc', simple_value=auc)])
            self._summary_writer.add_summary(summary, self._step)
        except tf.errors.OutOfRangeError:
            pass
        finally:
            coord.request_stop()
        coord.join(threads)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--output_root', type=str, required=True)
    parser.add_argument('--num_feature', type=int, required=True)
    args, _ = parser.parse_known_args()

    data_root = args.data_root
    output_root = args.output_root
    model = LinearRegressionCTR(data_root=data_root,
                                output_root=output_root,
                                num_feature=args.num_feature,
                                learning_rate=5e-4,
                                batch_size=512)
    for _ in xrange(10):
        model.train()
        model.save()
        model.validate()

if __name__ == '__main__':
    main()
