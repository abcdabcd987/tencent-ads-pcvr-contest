import os, zipfile, numpy as np, tensorflow as tf
from datetime import datetime
from src import utils, tfutils
from src.data import IndexRepresentation

class KeyMap:
	@classmethod
	def get(cls, k):
		if k is None:
			return cls.DEFAULT
		assert k in cls.MAP, "{} {} not found.".format(cls.NAME, k)
		return cls.MAP[k]

class Optimizers(KeyMap):
	NAME = "Optimizer"
	DEFAULT = tf.train.AdamOptimizer
	MAP = {
		"adam": tf.train.AdamOptimizer
	}

class ModelTemplate(object):
	def __init__(self, config):
		self._num_feature = config.dense_shape
		self._num_one = config.max_length
		self._learning_rate = config.learning_rate
		self._batch_size = config.batch_size
		self._optimizer = config.optimizer

		self._build()

		config.model = self

	def _body(self):
		w = tf.get_variable('weight', [self._num_feature], dtype=tf.float32,
							initializer=tf.random_uniform_initializer(-0.05, 0.05))
		b = tf.get_variable('bias', [1], dtype=tf.float32,
							initializer=tf.zeros_initializer())
		wx = tf.reduce_sum(tf.gather(w, self.ph_x), axis=1)
		return wx + b

	def _build(self):
		self.ph_x = tf.placeholder(tf.int32, [None, self._num_one])
		self.ph_y = tf.placeholder(tf.float32, [None])
		logits = self._body()
		self.prob = tf.sigmoid(logits)
		loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.ph_y, logits=logits))
		self.train_step = Optimizers.get(self._optimizer)(self._learning_rate).minimize(loss)
		avgloss, _ = tf.metrics.mean(loss, updates_collections=tf.GraphKeys.UPDATE_OPS)
		auc, _ = tf.metrics.auc(self.ph_y, self.prob, updates_collections=tf.GraphKeys.UPDATE_OPS)
		tf.summary.scalar("avg_loss", avgloss)
		tf.summary.scalar("auc", auc)
		self.summary_op = tf.summary.merge_all()
		self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

		self.global_step = tfutils.make_global_step()
		self.global_initializer = tf.global_variables_initializer()
		self.local_initializer = tf.local_variables_initializer()

		self.saver = tf.train.Saver()
