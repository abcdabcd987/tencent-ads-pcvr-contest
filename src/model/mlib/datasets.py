from src import data
from src.data import IndexRepresentation

class TestIterator:
	def __init__(self, iterator):
		self.iterator = iterator

	def __iter__(self):
		for xs, ids in self.iterator:
			yield ids, xs, None

class NormalIterator:
	def __init__(self, iterator):
		self.iterator = iterator

	def __iter__(self):
		for xs, ys in self.iterator:
			yield None, xs, ys

class Datasets:
	def __init__(self, config):
		self._data_storage = data.DataStorage(config.features)
		self._data_storage.load_data()
		print('feature data loaded')

		self._rep = self._data_storage.get_representation(IndexRepresentation)

		config.dense_shape = self._rep.dense_shape
		config.max_length = self._rep.max_length
		config.datasets = self

	def get_dataset(self, dataset, batch_size):
		if dataset == 'test':
			return TestIterator(self._rep.get_dataset(dataset, batch_size))
		else:
			return NormalIterator(self._rep.get_dataset(dataset, batch_size))
