class ModelConfig:
	def __init__(self, config):
		self._config = config

	@property
	def __getattr__(self, key):
		return self._config.__getattribute__(key)
