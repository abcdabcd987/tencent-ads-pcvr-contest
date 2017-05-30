import os, zipfile, numpy as np, tensorflow as tf
from datetime import datetime
from mlib import tfutils
from mlib.datareader import BufferedDataReader

class KeyMap:
	@classmethod
	def get(cls, k):
		assert k in cls.MAP, "{} {} not found.".format(cls.NAME, k)
		return cls.MAP[k]

class Initializers(KeyMap):
	NAME = "Initializer"
	MAP = {
		"adam": tf.train.AdamOptimizer
	}

class ModelTemplate(object):
	def __init__(self, **kwargs):
		self._data_root = kwargs['data_root']
		self._output_root = kwargs['output_root']
		self._num_one = kwargs['num_one']
		self._optimizer = kwargs['optimizer']
		self._learning_rate = kwargs['learning_rate']
		self._batch_size = kwargs['batch_size']

		self._build()

    def _body(self):
    	pass

	def _build(self):
		self._ph_x = tf.placeholder(tf.int32, [None, self._num_one], name="feed_x")
		self._ph_y = tf.placeholder(tf.float32, [None], name="feed_y")

		logits = self._body();
		self._global_step = tfutils.make_global_step()

		self._prob = tf.sigmoid(logits)
		self._loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self._ph_y, logits=logits))
		opt = Initializers.get(self._optimizer)(self._learning_rate, global_step=self._global_step)
		self._train_step = opt.minimize(self._loss)
		self._auc, self._auc_op = tf.metrics.auc(self._ph_y, self._prob)

		self._session_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_' + self.__class__.MODEL_NAME
		self._model_dir = os.path.join(self._output_root, 'models', self._session_name)
		os.makedirs(self._model_dir)
		log_dir = os.path.join(self._output_root, 'logs', self._session_name)
		train_log_dir = os.path.join(log_dir, "train")
		os.makedirs(train_log_dir)
		self._train_summary_writer = tf.summary.FileWriter(train_log_dir, self._sess.graph, flush_secs=2)
		val_log_dir = os.path.join(log_dir, "val")
		os.makedirs(val_log_dir)
		self._val_summary_writer = tf.summary.FileWriter(val_log_dir, flush_secs=2)

		self._saver = tf.train.Saver()
		self._sess = tfutils.create_session()

		self._global_initializer = tf.global_variables_initializer()
		self._local_initializer = tf.local_variables_initializer()

		self._sess.run(self._global_initializer)
		self._step = 0
	
	def load(self, model_path):
		tfutils.load_session(self._sess, self._saver, model_path)

	def save(self):
		tfutils.save_session(self._sess, self._saver, model_path, self._step)

	def train(self):
		filename = os.path.join(self._data_root, 'train.txt.gz')
		reader = BufferedDataReader(filename, self._batch_size, self._num_one)
		self._sess.run(self._local_initializer)
		try:
			while True:
				xs, ys = reader.get_batch()
				if len(ys) == 0:
					break
				_, _, loss, auc, self._step = self._sess.run([self._train_step, self._auc_op,
															  self._loss, self._auc, self._global_step],
															 feed_dict={self._ph_x: xs, self._ph_y: ys})
				if self._step % 16 == 0:
					self._sess.run(tf.local_variables_initializer())
					_, num_batch_buffer = reader.qsize()
					summary = tf.Summary(value=[tf.Summary.Value(tag="loss", simple_value=loss),
												tf.Summary.Value(tag="auc", simple_value=auc),
												tf.Summary.Value(tag="batch_buffer", simple_value=num_batch_buffer)])
					self._summary_writer.add_summary(summary, self._step)
		finally:
			reader.stop()
			reader.join()

	def validate(self):
		valsess = tfutils.create_session()
		valsess.run(self._global_initializer)
		tfutils.load_session(valsess, self._saver, self._model_dir)
		filename = os.path.join(self._data_root, 'val.txt.gz')
		reader = BufferedDataReader(filename, self._batch_size, self._num_one)
		self._sess.run(self._local_initializer)
		try:
			while True:
				xs, ys = reader.get_batch()
				if len(ys) == 0:
					break
				self._sess.run(self._auc_op, feed_dict={self._ph_x: xs, self._ph_y: ys})
			auc = self._sess.run(self._auc)
			summary = tf.Summary(value=[tf.Summary.Value(tag='val_auc', simple_value=auc)])
			self._summary_writer.add_summary(summary, self._step)
		finally:
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
		finally:
			reader.stop()
			reader.join()
		res.sort(key=lambda (i, prob): i)

		test_dir = os.path.join(self._output_root, 'tests')
		if not os.path.exists(test_dir):
			os.makedirs(test_dir)
		filename = os.path.join(test_dir, self._session_name)
		with open(filename + '.csv', 'w') as f:
			f.write('instanceID,prob\n')
			for i, prob in res:
				f.write('{:d},{:.16f}\n'.format(i, prob))
		with zipfile.ZipFile(filename + '.zip', 'w', zipfile.ZIP_DEFLATED) as z:
			z.write(filename + '.csv', 'submission.csv')
		print('test result wrote to', filename + '.csv')