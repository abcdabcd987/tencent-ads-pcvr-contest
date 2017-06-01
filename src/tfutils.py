import numpy as np, tensorflow as tf, os

def make_global_step():
	return tf.get_variable("global_step", (), dtype=tf.int64,
						   initializer=tf.zeros_initializer(),
						   collections=[tf.GraphKeys.GLOBAL_VARIABLES,
										tf.GraphKeys.GLOBAL_STEP])

def create_session():
	config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
	config.gpu_options.allow_growth=True
	return tf.Session(config=config)

def load_session(sess, saver, model_path):
	checkpoint = tf.train.get_checkpoint_state(model_path)
	if checkpoint and checkpoint.model_checkpoint_path:
		saver.restore(sess, checkpoint.model_checkpoint_path)
		print("model loaded:", checkpoint.model_checkpoint_path)
	else:
		raise Exception("no model found in " + model_path)

def save_session(sess, saver, model_path, step):
	filename = os.path.join(model_path, 'model')
	saver.save(sess, filename, global_step=step)
