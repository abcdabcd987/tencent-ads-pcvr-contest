import os, os.path

class ExpDirectories:
	def __init__(self, models_dir, logs_dir, tests_dir, exp_name):
		self.model_dir = os.path.join(models_dir, exp_name)
		self.log_dir = os.path.join(logs_dir, exp_name)
		self.test_dir = os.path.join(tests_dir, exp_name)

	def session(self, sessid):
		sess_dirs = SessionDirectories(self, sessid)
		sess_dirs.makedirs()
		return sess_dirs

class SessionDirectories:
	def __init__(self, exp_directories, sessid):
		self.model_dir = os.path.join(exp_directories.model_dir, sessid)
		self.log_dir = os.path.join(exp_directories.log_dir, sessid)
		self.test_dir = os.path.join(exp_directories.test_dir, sessid)

		self.train_log_dir = os.path.join(self.log_dir, "train")
		self.val_log_dir = os.path.join(self.log_dir, "val")

	def makedirs(self):
		os.makedirs(self.model_dir)
		os.makedirs(self.train_log_dir)
		os.makedirs(self.val_log_dir)
		os.makedirs(self.test_dir)
