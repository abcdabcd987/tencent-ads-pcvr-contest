import warnings
import numpy as np
from .RepresentationBase import *

class IndexBatchIterator(FeatureBatchIterator):
    def __next__(self):
        s = self._r.storage
        if not s.data_loaded:
            raise RuntimeError('Datastorage has not loaded feature data yet.')
        indexes = self._get_batch_perm()
        xs = np.zeros((len(indexes), self._r.max_length), dtype=np.int32)
        ys = np.array(s.feature_labels[indexes], copy=True, dtype=np.int32)
        rowids = indexes
        for i, index in enumerate(indexes):
            j = 0
            for meta, data, offset in zip(s.feature_metas, s.feature_data, self._r.dimension_offsets):
                if meta['type'] == 'one_hot':
                    xs[i][j] = offset + data[index]
                    assert j < self._r.max_length
                    j += 1
                elif meta['type'] == 'multi_hot':
                    idx, val = data
                    for k in idx[index]:
                        xs[i][j] = offset + k
                        j += 1
                else:
                    raise NotImplementedError('IndexBatchIterator forgets to support feature type: ' + repr(meta['type']))
        return xs, ys, rowids


class IndexRepresentation(FeatureRepresentationBase):
    _multihot_warned = False

    def _setup(self):
        if not IndexRepresentation._multihot_warned:
            msg = 'IndexRepresentation is a index-only representation. Values of multi-hot features will be ignored.'
            warnings.warn(msg)
            IndexRepresentation._multihot_warned = True

        self._offsets = []
        self._dense_shape = 0
        self._max_length = 0
        for name, meta in zip(self._storage.feature_names, self._storage.feature_metas):
            if meta['type'] == 'one_hot':
                dimension = meta['dimension']
                self._max_length += 1
            elif meta['type'] == 'multi_hot':
                dimension = meta['dimension']
                self._max_length += meta['max_length']
            else:
                raise TypeError('IndexRepresentation does not support feature type: ' + repr(meta['type']))
            self._offsets.append(self._dense_shape)
            self._dense_shape += dimension
    
    @property
    def dimension_offsets(self):
        return self._offsets

    @property
    def dense_shape(self):
        return self._dense_shape

    @property
    def max_length(self):
        return self._max_length

    @property
    def iterator_class(self):
        return IndexBatchIterator
