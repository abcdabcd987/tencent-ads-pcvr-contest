from datetime import datetime
from ..directories import ExpDirectories

class ExpConfig:
	def __init__(self, config):
		self._config = config
		exp_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + config.module_name
		self._exp_directories = ExpDirectories(config.models_dir, config.logs_dir, config.tests_dir, exp_name)

	def session(self, sessid):
		return SessionConfig(self._config, self._exp_directories, sessid)

class SessionConfig:
	def __init__(self, config, exp_directories, sessid):
		self._config = config
		self.sess_dirs = exp_directories.session(str(sessid))

		if isinstance(sessid, int):
			self.train = 'smalltrain{}'.format(sessid)
			self.val = 'val{}'.format(sessid)
			self.test = None
		else:
			self.train = 'train'
			self.val = None
			self.test = 'test'

	def __getattr__(self, key):
		return eval("self._config.{}".format(key))
