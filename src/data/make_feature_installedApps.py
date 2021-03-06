import argparse
import os
import glob
import math
from array import array
from collections import namedtuple
from tqdm import tqdm

from .utils import *
from ..utils import *


def make_installedApps(other_threshold=4):
    global config
    installedApps_list = load_feature(os.path.join(config['features_dir'], 'raw', 'installedApps.npy'))

    all_apps = array('l')
    for apps in tqdm(installedApps_list):
        all_apps.extend(apps)
    count = count_values(all_apps)
    index = index_values(count, all_apps, other_threshold)

    max_length = 0
    res_idx = []
    res_val = []
    for apps in tqdm(installedApps_list):
        idx = remap_feature(index, apps)
        idx.sort()
        max_length = max(max_length, len(idx))
        # we want: sqrt(len(idx) * value^2) == len(idx) / max_length
        value = math.sqrt(len(idx)) / max_length
        val = np.ones(len(idx), dtype=np.float32) * value
        res_idx.append(idx)
        res_val.append(val)

    dump_meta(os.path.join(config['features_dir'], 'basic', 'installedApps.meta.json'),
              {'type': 'multi_hot', 'dimension': len(index),
               'max_length': max_length, 'index': index, 'count': count})
    dump_feature(os.path.join(config['features_dir'], 'basic', 'installedApps.npy'),
                 res_idx, res_val)
    print('done multi-hot feature: installedApps')


def main():
    global config
    config = read_global_config()

    make_installedApps(other_threshold=4)


main()
