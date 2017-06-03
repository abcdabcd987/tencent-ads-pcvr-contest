import os, zipfile, tensorflow as tf
from src import utils, tfutils

class Session(object):
	def __init__(self, config):
		self._datasets = config.datasets
		self._model = config.model
		self._model_dir = config.sess_dirs.model_dir
		self._train_log_dir = config.sess_dirs.train_log_dir
		self._val_log_dir = config.sess_dirs.val_log_dir
		self._test_dir = config.sess_dirs.test_dir
		self._batch_size = config.batch_size

		self._start()

		self._train_dataset = config.train_dataset
		self._val_dataset = config.val_dataset
		self._test_dataset = config.test_dataset

	def _start(self):
		self._ph_x = self._model.ph_x
		self._ph_y = self._model.ph_y
		self._prob = self._model.prob
		self._train_step = self._model.train_step
		self._summary_op = self._model.summary_op
		self._update_ops = self._model.update_ops

		self._global_step = self._model.global_step
		self._global_initializer = self._model.global_initializer
		self._local_initializer = self._model.local_initializer

		self._saver = self._model.saver
		self._sess = tfutils.create_session()
		self._train_sum_writer = tf.summary.FileWriter(self._train_log_dir, self._sess.graph, flush_secs=2)
		self._val_sum_writer = tf.summary.FileWriter(self._val_log_dir, flush_secs=2)

		self._sess.run(self._global_initializer)
		self._step = self._sess.run(self._global_step)

	def _get_dataset(self, dataset):
		return self._datasets.get_dataset(dataset, self._batch_size)

	def _recover_sess(self):
		sess = tfutils.create_session()
		sess.run(self._global_initializer)
		tfutils.save_session(sess, self._saver, self._model_dir, self._step)
		tfutils.load_session(sess, self._saver, self._model_dir)
		return sess

	def load(self, model_path):
		tfutils.load_session(self._sess, self._saver, model_path)

	def save(self):
		tfutils.save_session(self._sess, self._saver, self._model_dir, self._step)

	def train(self, dataset):
		self._sess.run(self._local_initializer)
		for _, xs, ys in self._get_dataset(dataset):
			_, _, self._step = self._sess.run([self._train_step, self._update_ops, self._global_step],
											  feed_dict={self._ph_x: xs, self._ph_y: ys})
			if self._step % 16 == 0:
				self._sess.run(self._local_initializer)
				summary = self._sess.run(self._summary_op)
				self._train_sum_writer.add_summary(summary, self._step)
				self._train_sum_writer.flush()

	def validate(self, dataset):
		sess = self._recover_sess()
		sess.run(self._local_initializer)
		for _, xs, ys in self._get_dataset(dataset):
			sess.run(self._update_ops, feed_dict={self._ph_x: xs, self._ph_y: ys})
		summary = sess.run(self._summary_op)
		self._val_sum_writer.add_summary(summary, self._step)
		self._val_sum_writer.flush()
		sess.close()

	def test(self, dataset):
		sess = self._recover_sess()
		res = []
		for ids, xs, _ in self._get_dataset(dataset):
			probs = sess.run(self._prob, feed_dict={self._ph_x: xs})
			res.extend([(i, prob) for i, prob in zip(ids, probs)])
		res.sort(key=lambda v: v[0])

		filename = os.path.join(self._test_dir, 'result')
		utils.write_zip(filename, res)
