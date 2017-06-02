from src import utils
from .data_config import DataConfig
from .model_config import ModelConfig
from .session_config import ExpConfig

class Config:
	def __init__(self):
		self.__internal__ = {}
		self.data_config = DataConfig(self)
		self.model_config = ModelConfig(self)
		self.exp_config = ExpConfig(self)

	@classmethod
	def from_json(cls, script_file, config_name):
		config = utils.read_module_config(script_file, config_name)
		instance = cls()
		instance.__internal__.update(config)

	@property
	def dense_shape(self):
		return self.data_config.dense_shape

	@property
	def max_length(self):
		return self.data_config.max_length

	@property
	def datasets(self):
		return self.data_config.datasets

	@property
	def model(self):
		return self.model_config.model

	def __getattr__(self, key):
		return self.__internal__.get(key)

	def session_config(self, sessid):
		return self.exp_config.session(sessid)
