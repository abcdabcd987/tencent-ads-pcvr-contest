import os
import json
import cPickle

def get_num_lines(filename):
    with open(filename) as f:
        return sum(1 for line in f)


def dump_feature(filename, content):
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    with open(filename, 'wb') as f:
        cPickle.dump(content, f, cPickle.HIGHEST_PROTOCOL)


def dump_meta(filename, meta):
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    with open(filename, 'wb') as f:
        json.dump(meta, f, sort_keys=True, indent=2)


def load_feature(filename):
    with open(filename, 'rb') as f:
        return cPickle.load(f)


def load_meta(filename):
    with open(filename, 'rb') as f:
        return json.load(f)


def count_values(values):
    count = {}
    for value in values:
        count[value] = count.get(value, 0) + 1
    return count


def index_values(count, values, threshold):
    l = sorted(count.iteritems(), key=lambda (value, cnt): value)
    m = {'__other__': 0}
    for category, count in l:
        if count >= threshold:
            m[category] = len(m)
    return m


def remap_feature(feature_map, values):
    feature = []
    for value in values:
        idx = feature_map.get(value, None)
        if idx is None:
            idx = feature_map['__other__']
        feature.append(idx)
    return feature
