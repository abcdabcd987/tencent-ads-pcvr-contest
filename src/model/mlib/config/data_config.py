class DataConfig:
	def __init__(self, config):
		self._config = config

	@property
	def features(self):
		return self._config.features
