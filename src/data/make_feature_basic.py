#!/usr/bin/env python2

import os
import math
import numpy as np
from array import array
from collections import namedtuple
from tqdm import tqdm

from .utils import *
from ..utils import *


def make_onehot_feature(name, other_threshold=4):
    global config

    values = load_feature(os.path.join(config['features_dir'], 'raw', name + '.npy'))
    count = count_values(values)
    index = index_values(count, values, other_threshold)
    res = remap_feature(index, values)

    meta = {'type': 'one_hot', 'dimension': len(index), 'index': index, 'count': count}
    dump_meta(os.path.join(config['features_dir'], 'basic', name + '.meta.json'),
              {'type': 'one_hot', 'dimension': len(index), 'index': index, 'count': count})
    dump_feature(os.path.join(config['features_dir'], 'basic', name + '.npy'), res)
    print('done one-hot feature:', name)


def main():
    global config
    config = read_global_config()

    # ad
    make_onehot_feature('creativeID')
    make_onehot_feature('adID')
    make_onehot_feature('camgaignID')
    make_onehot_feature('advertiserID')
    make_onehot_feature('appPlatform')

    # app
    make_onehot_feature('appID')
    make_onehot_feature('appCategory')

    # user
    make_onehot_feature('userID')
    make_onehot_feature('age')
    make_onehot_feature('gender')
    make_onehot_feature('education')
    make_onehot_feature('marriageStatus')
    make_onehot_feature('haveBaby')
    make_onehot_feature('hometown')
    make_onehot_feature('residence')

    # context
    # make_onehot_feature('clickTime') # we don't make clickTime directly into one-hot
    make_onehot_feature('positionID')
    make_onehot_feature('sitesetID')
    make_onehot_feature('positionType')
    make_onehot_feature('connectionType')
    make_onehot_feature('telecomsOperator')

main()
