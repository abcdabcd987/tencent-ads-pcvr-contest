#!/usr/bin/env python2

import argparse
import os
import math
from collections import namedtuple
from tqdm import tqdm
from array import array

from .utils import *
from ..utils import *


def make_clickWeekday_clickHour():
    global config
    clickTimes = load_feature(os.path.join(config['features_dir'], 'raw', 'clickTime.npy'))
    clickWeekday = np.empty(len(clickTimes), dtype=np.int32)
    clickHour = np.empty(len(clickTimes), dtype=np.int32)
    for i, clickTime in enumerate(clickTimes):
        dd, hh, mm = clickTime / 10000, clickTime / 100 % 100, clickTime % 100
        clickWeekday[i] = dd % 7
        clickHour[i] = hh
    meta = {'type': 'numeric', 'dimension': 1}
    dump_meta(os.path.join(config['features_dir'], 'extend', 'clickWeekday.meta.json'), meta)
    dump_feature(os.path.join(config['features_dir'], 'extend', 'clickWeekday.pkl'), clickWeekday)
    dump_meta(os.path.join(config['features_dir'], 'extend', 'clickHour.meta.json'), meta)
    dump_feature(os.path.join(config['features_dir'], 'extend', 'clickHour.pkl'), clickHour)


def main():
    global config
    config = read_global_config()
    actions = [
        make_clickWeekday_clickHour,
    ]

    actions = {f.__name__: f for f in actions}
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=list(actions.keys()))
    args = parser.parse_args()
    actions[args.action]()
    print('done')


main()
