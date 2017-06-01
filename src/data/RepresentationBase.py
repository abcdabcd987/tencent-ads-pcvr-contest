import abc
import numpy as np


class FeatureBatchIterator(metaclass=abc.ABCMeta):
    def __init__(self, representation, dataset, batch_size, allow_smaller_final_batch):
        self._r = representation
        self._dataset = dataset
        self._batch_size = batch_size
        self._allow_smaller_final_batch = allow_smaller_final_batch

        self._slice = self._r.storage.dataset_slices[self._dataset]
        self._perm = np.random.permutation(np.arange(self._slice.start, self._slice.stop))
        self._perm_head = 0

    def _get_batch_perm(self):
        if self._perm_head >= len(self._perm):
            raise StopIteration
        stop = self._perm_head + self._batch_size
        if stop > len(self._perm) and not self._allow_smaller_final_batch:
            self._perm_head = len(self._perm)
            raise StopIteration
        batch = self._perm[self._perm_head:stop]
        self._perm_head = stop
        return batch
    
    def __iter__(self):
        return self

    @abc.abstractmethod
    def next(self):
        pass


class FeatureRepresentationBase(metaclass=abc.ABCMeta):
    def __init__(self, data_storage):
        self._storage = data_storage
        self._setup()

    def get_dataset(self, dataset, batch_size, allow_smaller_final_batch=True):
        if dataset not in self._storage.dataset_slices:
            raise KeyError('Dataset ' + repr(dataset) + ' not found')
        iterator = self.iterator_class
        if not issubclass(iterator, FeatureBatchIterator):
            raise TypeError(repr(iterator) + ' is not subclass of FeatureBatchIterator')
        return iterator(self, dataset, batch_size, allow_smaller_final_batch)
    
    @property
    def storage(self):
        return self._storage
    
    @abc.abstractmethod
    def _setup(self):
        pass
    
    @abc.abstractproperty
    def iterator_class(self):
        pass
