import os
import json
import cPickle
import itertools
import numpy as np
from array import array

def get_num_lines(filename):
    with open(filename) as f:
        return sum(1 for line in f)


def dump_feature(filename, *arrays):
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    arr = arrays[0] if isinstance(arrays, tuple) and len(arrays) == 1 else arrays
    np.save(filename, arr)


def dump_meta(filename, meta):
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    with open(filename, 'wb') as f:
        json.dump(meta, f, sort_keys=True, indent=2)


def load_feature(filename):
    return np.load(filename)


def load_meta(filename):
    with open(filename, 'rb') as f:
        return json.load(f)


def count_values(values):
    count = {}
    for value in values:
        value = int(value)
        count[value] = count.get(value, 0) + 1
    return count


def index_values(count, values, threshold):
    l = sorted(count.iteritems(), key=lambda (value, cnt): value)
    m = {'__other__': 0}
    for category, count in l:
        if count >= threshold:
            m[category] = len(m)
    return m


def remap_feature(feature_map, values, dtype=np.int32):
    feature = np.empty(len(values), dtype=dtype)
    assert len(feature_map) < np.iinfo(dtype).max
    for i, value in enumerate(values):
        idx = feature_map.get(value, None)
        if idx is None:
            idx = feature_map['__other__']
        feature[i] = idx
    return feature
