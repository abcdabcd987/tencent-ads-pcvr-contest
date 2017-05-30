from __future__ import print_function
import os
import abc
import sys
import warnings
import numpy as np

from .utils import *
from ..utils import *


class BatchRepresentationBase:
    __metaclass__ = abc.ABCMeta

    def __init__(self, data_reader, dataset, batch_size, allow_smaller_final_batch):
        self._reader = data_reader
        self._dataset = dataset
        self._batch_size = batch_size
        self._allow_smaller_final_batch = allow_smaller_final_batch

        self._slice = self._reader.dataset_slices[self._dataset]
        self._perm = np.random.permutation(np.arange(self._slice.start, self._slice.stop))
        self._perm_head = 0
    
    def _get_batch_perm(self):
        if self._perm_head == len(self._perm):
            raise StopIteration
        batch = self._perm[self._perm_head : self._perm_head + self._batch_size]
        self._perm_head += self._batch_size
        return batch


class IndexBatch(BatchRepresentationBase):
    _multihot_warned = False

    def __init__(self, *args):
        """
        :param multihot_action accept / warn / reject
        """
        super(IndexBatch, self).__init__(*args)
        if not IndexBatch._multihot_warned:
            msg = 'BatchRepresentationBase is a index-only representation. Values of multi-hot features will be ignored.'
            warnings.warn(msg)
            IndexBatch._multihot_warned = True

        self._dimension_offsets = []
        self._dense_shape = 0
        self._max_length = 0
        for name, meta in zip(self._reader.feature_names, self._reader.feature_metas):
            if meta['type'] == 'one_hot':
                dimension = meta['dimension']
                self._max_length += 1
            elif meta['type'] == 'multi_hot':
                dimension = meta['dimension']
                self._max_length += meta['max_length']
            else:
                raise TypeError('IndexBatch does not support feature type: ' + repr(meta['type']))
            self._dimension_offsets.append(self._dense_shape)
            self._dense_shape += dimension
    
    def __iter__(self):
        return self
    
    def next(self):
        if not self._reader.data_loaded:
            raise RuntimeError('DataReader has not loaded feature data yet.')
        indexes = self._get_batch_perm()
        xs = np.zeros((len(indexes), self._max_length), dtype=np.int32)
        ys = np.array(self._reader.feature_labels[indexes], copy=True, dtype=np.int32)
        for i, index in enumerate(indexes):
            j = 0
            for meta, data, offset in zip(self._reader.feature_metas, self._reader.feature_data, self._dimension_offsets):
                if meta['type'] == 'one_hot':
                    xs[i][j] = offset + data[index]
                    j += 1
                elif meta['type'] == 'multi_hot':
                    idx, val = data
                    for k in idx[index]:
                        xs[i][j] = offset + k
                        j += 1
                else:
                    raise NotImplementedError('IndexBatch forgets to support feature type: ' + repr(meta['type']))
        return xs, ys
    
    @property
    def dense_shape(self):
        return self._dense_shape

    @property
    def max_length(self):
        return self._max_length


class DataReader(object):
    def __init__(self, features):
        config = read_global_config()
        self._data_loaded = False
        self._features_dir = config['features_dir']
        self._feature_names = features
        self._load_metas()
        self._partition_datasets()
    
    def _load_metas(self):
        self._feature_metas = []
        for name in self._feature_names:
            meta = load_meta(os.path.join(self._features_dir, name + '.meta.json'))
            self._feature_metas.append(meta)

    def load_data(self):
        self._feature_data = []
        for name in self._feature_names:
            data = load_feature(os.path.join(self._features_dir, name + '.npy'))
            self._feature_data.append(data)
        self._labels = load_feature(os.path.join(self._features_dir, 'label.npy'))
        self._data_loaded = True
    
    def _partition_datasets(self):
        def do(out, clickTimes, date_st, date_ed, name):
            st = np.searchsorted(clickTimes, date_st * 10000)
            ed = np.searchsorted(clickTimes, (date_ed + 1) * 10000)
            out[name] = slice(st, ed)
        clickTimes = load_feature(os.path.join(self._features_dir, 'raw', 'clickTime.npy'))
        self._datasets = {}
        do(self._datasets, clickTimes, 17, 23, 'smalltrain1')
        do(self._datasets, clickTimes, 24, 24, 'val1')
        do(self._datasets, clickTimes, 17, 27, 'smalltrain2')
        do(self._datasets, clickTimes, 28, 28, 'val2')
        do(self._datasets, clickTimes, 17, 28, 'smalltrain3')
        do(self._datasets, clickTimes, 29, 29, 'val3')
        do(self._datasets, clickTimes, 17, 29, 'smalltrain4')
        do(self._datasets, clickTimes, 30, 30, 'val4')
        do(self._datasets, clickTimes, 17, 30, 'train')
        do(self._datasets, clickTimes, 31, 31, 'test')
    
    def get_dataset(self, dataset, batch_representation, batch_size, allow_smaller_final_batch=True):
        if dataset not in self._datasets:
            raise KeyError('Dataset ' + repr(dataset) + ' not found')
        return batch_representation(self, dataset, batch_size, allow_smaller_final_batch)
        

    @property
    def feature_names(self):
        return self._feature_names

    @property
    def feature_metas(self):
        return self._feature_metas

    @property
    def feature_data(self):
        return self._feature_data
    
    @property
    def feature_labels(self):
        return self._labels
    
    @property
    def dataset_slices(self):
        return self._datasets
    
    @property
    def data_loaded(self):
        return self._data_loaded
