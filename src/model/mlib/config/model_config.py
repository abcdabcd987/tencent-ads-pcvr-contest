class ModelConfig:
	def __init__(self, config):
		self._config = config

	def __getattr__(self, key):
		return eval("self._config.{}".format(key))
