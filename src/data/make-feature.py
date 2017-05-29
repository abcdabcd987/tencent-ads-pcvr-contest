#!/usr/bin/env python2

import argparse
import os
import math
from collections import namedtuple
from tqdm import tqdm

from utils import *


def make_clickWeekday_clickHour(args):
    clickTimes = load_feature(os.path.join(args.feature_dir, 'raw', 'clickTime.pkl'))
    clickWeekday = []
    clickHour = []
    for clickTime in clickTimes:
        dd, hh, mm = clickTime / 10000, clickTime / 100 % 100, clickTime % 100
        clickWeekday.append(dd % 7)
        clickHour.append(hh)
    meta = {'type': 'numeric', 'dimension': 1}
    dump_meta(os.path.join(args.feature_dir, 'extend', 'clickWeekday.meta.json'), meta)
    dump_feature(os.path.join(args.feature_dir, 'extend', 'clickWeekday.pkl'), clickWeekday)
    dump_meta(os.path.join(args.feature_dir, 'extend', 'clickHour.meta.json'), meta)
    dump_feature(os.path.join(args.feature_dir, 'extend', 'clickHour.pkl'), clickHour)


if __name__ == '__main__':
    actions = [
        make_clickWeekday_clickHour,
    ]

    actions = {f.__name__: f for f in actions}
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=actions.keys())
    parser.add_argument('--feature_dir', type=str, required=True)
    args = parser.parse_args()
    actions[args.action](args)
    print 'done'
    
