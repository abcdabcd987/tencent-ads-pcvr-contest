import os, zipfile, numpy as np, tensorflow as tf
from datetime import datetime
from src import utils, tfutils
from src.data import IndexRepresentation

class KeyMap:
	@classmethod
	def get(cls, k):
		assert k in cls.MAP, "{} {} not found.".format(cls.NAME, k)
		return cls.MAP[k]

class Optimizers(KeyMap):
	NAME = "Optimizer"
	MAP = {
		"adam": tf.train.AdamOptimizer
	}

class ModelTemplate(object):
	def __init__(self, data_storage, config):
		self._rep = data_storage.get_representation(IndexRepresentation)
		self._num_feature = self._rep.dense_shape
		self._num_one = self._rep.max_length
		self._learning_rate = config['learning_rate']
		self._batch_size = config['batch_size']
		self._optimizer = config.get('initializer', 'adam')
		self._session_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_' + config['module_name']
		self._model_dir = os.path.join(config['models_dir'], self._session_name)
		self._log_dir = os.path.join(config['logs_dir'], self._session_name)
		self._train_log_dir = os.path.join(self._log_dir, "train")
		self._val_log_dir = os.path.join(self._log_dir, "val")
		self._test_dir = config['tests_dir']

		self._makefolders()

		self._build()

	def _makefolders(self):
		os.makedirs(self._model_dir)
		os.makedirs(self._log_dir)
		os.makedirs(self._train_log_dir)
		os.makedirs(self._val_log_dir)
		if not os.path.exists(self._test_dir):
			os.makedirs(self._test_dir)

	def _body(self):
		w = tf.get_variable('weight', [self._num_feature], dtype=tf.float32,
							initializer=tf.random_uniform_initializer(-0.05, 0.05))
		b = tf.get_variable('bias', [1], dtype=tf.float32,
							initializer=tf.zeros_initializer())
		wx = tf.reduce_sum(tf.gather(w, self._ph_x), axis=1)
		return wx + b

	def _build(self):
		self._ph_x = tf.placeholder(tf.int32, [None, self._num_one])
		self._ph_y = tf.placeholder(tf.float32, [None])
		logits = self._body()
		self._prob = tf.sigmoid(logits)
		loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self._ph_y, logits=logits))
		self._train_step = Optimizers.get(self._optimizer)(self._learning_rate).minimize(loss)
		avgloss, _ = tf.metrics.mean(loss, updates_collections=tf.GraphKeys.UPDATE_OPS)
		auc, _ = tf.metrics.auc(self._ph_y, self._prob, updates_collections=tf.GraphKeys.UPDATE_OPS)
		tf.summary.scalar("avg_loss", avgloss)
		tf.summary.scalar("auc", auc)
		self._summary_op = tf.summary.merge_all()
		self._update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

		self._global_step = tfutils.make_global_step()
		self._global_initializer = tf.global_variables_initializer()
		self._local_initializer = tf.local_variables_initializer()

		self._saver = tf.train.Saver()
		self._sess = tfutils.create_session()
		self._train_sum_writer = tf.summary.FileWriter(self._train_log_dir, self._sess.graph, flush_secs=2)
		self._val_sum_writer = tf.summary.FileWriter(self._val_log_dir, flush_secs=2)

		self._sess.run(self._global_initializer)
		self._step = 0

	def _get_dataset(self, dataset):
		return self._rep.get_dataset(dataset, self._batch_size)

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
		self._sess.run(tf.local_variables_initializer())
		for xs, ys in self._get_dataset(dataset):
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
		for xs, ys in self._get_dataset(dataset):
			sess.run(self._update_ops, feed_dict={self._ph_x: xs, self._ph_y: ys})
		summary = sess.run(self._summary_op)
		self._val_sum_writer.add_summary(summary, self._step)
		sess.close()

	def test(self, dataset):
		sess = self._recover_sess()
		res = []
		for xs, ys in self._get_dataset(dataset):
			probs = sess.run(self._prob, feed_dict={self._ph_x: xs})
			res.extend([(i, prob) for i, prob in zip(ys, probs)])
		res.sort(key=lambda v: v[0])

		filename = os.path.join(self._test_dir, self._session_name)
		utils.write_zip(filename, res)
