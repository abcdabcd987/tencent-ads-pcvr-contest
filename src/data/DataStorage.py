from __future__ import print_function
import os
import abc
import sys
import numpy as np

from .utils import *
from .RepresentationBase import *
from ..utils import *


class DataStorage(object):
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
    
    def get_representation(self, representation_class):
        if not issubclass(representation_class, FeatureRepresentationBase):
            raise TypeError(repr(representation_class) + ' is not subclass of FeatureRepresentationBase')
        return representation_class(self)

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
