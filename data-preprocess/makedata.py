#!/usr/bin/env python2

import argparse
import csv
import gzip
import sqlite3
import os
import multiprocessing
from collections import namedtuple, defaultdict
import numpy as np
from tqdm import tqdm

def get_ddhhmm(s):
    return s[:2], s[2:4], s[4:]

def get_app_list(userID, time):
    global cursor
    cursor.execute('''SELECT appID FROM user_installedapps WHERE userID={userID} UNION
        SELECT appID FROM user_app_actions WHERE userID={userID} AND installTime<={installTime}
        '''.format(userID=userID, installTime=time))
    return [x[0] for x in cursor.fetchall()]

class FeaturePair(namedtuple('FeaturePairBase', ['field', 'category'])):
    def __str__(self):
        return '{}::{}'.format(self.field, self.category)

    @staticmethod
    def from_row(row, field):
        return FeaturePair(field, row[field])

Impression = namedtuple('Impression', ['features', 'label', 'clickTime'])

def make_impression(row):
    global cursor
    features = []
    label = row['label']
    clickTime = row['clickTime']

    dd, hh, mm = get_ddhhmm(row['clickTime'])
    features.append(FeaturePair('clickWeekday', int(dd) % 7))
    features.append(FeaturePair('clickHour', int(hh)))

    cursor.execute('SELECT * FROM ads WHERE creativeID=' + row['creativeID'])
    c = cursor.fetchone()
    features.append(FeaturePair.from_row(c, 'creativeID'))
    features.append(FeaturePair.from_row(c, 'adID'))
    features.append(FeaturePair.from_row(c, 'camgaignID'))
    features.append(FeaturePair.from_row(c, 'advertiserID'))
    features.append(FeaturePair.from_row(c, 'appPlatform'))

    cursor.execute('SELECT * FROM app_categories WHERE appID=' + str(c['appID']))
    a = cursor.fetchone()
    features.append(FeaturePair.from_row(a, 'appID'))
    features.append(FeaturePair.from_row(a, 'appCategory'))

    cursor.execute('SELECT * FROM users WHERE userID=' + row['userID'])
    u = cursor.fetchone()
    features.append(FeaturePair.from_row(u, 'userID'))
    features.append(FeaturePair.from_row(u, 'age'))
    features.append(FeaturePair.from_row(u, 'gender'))
    features.append(FeaturePair.from_row(u, 'education'))
    features.append(FeaturePair.from_row(u, 'marriageStatus'))
    features.append(FeaturePair.from_row(u, 'haveBaby'))
    features.append(FeaturePair.from_row(u, 'hometown'))
    features.append(FeaturePair.from_row(u, 'residence'))
    for appID in get_app_list(row['userID'], row['clickTime']):
        features.append(FeaturePair('installedApp', appID))

    cursor.execute('SELECT * FROM positions WHERE positionID=' + row['positionID'])
    p = cursor.fetchone()
    features.append(FeaturePair.from_row(p, 'positionID'))
    features.append(FeaturePair.from_row(p, 'sitesetID'))
    features.append(FeaturePair.from_row(p, 'positionType'))

    features.append(FeaturePair.from_row(row, 'connectionType'))
    features.append(FeaturePair.from_row(row, 'telecomsOperator'))

    return features, label, clickTime


def make_feature_map():
    global train, feature_map, num_features
    count = {}
    fields = set()
    for imp in tqdm(train, total=len(train)):
        for feature in imp.features:
            count[feature] = count.get(feature, 0) + 1
            fields.add(feature.field)
    
    next_index = 0
    feature_map = {}
    feature_map[FeaturePair('__other__', '__other__')] = next_index
    next_index += 1
    for field in fields:
        feature_map[FeaturePair(field, '__other__')] = next_index
        next_index += 1

    for feature, count in count.iteritems():
        if count >= 10:
            feature_map[feature] = next_index
            next_index += 1
    num_features = next_index


def write_line(f, imp):
    f.write(imp.label + ' ')
    for feature in imp.features:
        idx = feature_map.get(feature, None)
        if idx is None:
            idx = feature_map[FeaturePair(feature.field, '__other__')]
        f.write(str(idx) + ':1 ')
    f.write('\n')


def setup_cursor(dbpath):
    global cursor
    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()


def get_num_lines(filename):
    with open(filename) as f:
        return sum(1 for line in f)


def main(args):
    global train, feature_map, cursor

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    setup_cursor(args.db)
    pool = multiprocessing.Pool(processes=args.num_workers, initializer=setup_cursor, initargs=(args.db, ))

    print 'reading train to build features...'
    num_ones = 0
    train = []
    filename = os.path.join(args.input_dir, 'train.csv')
    with open(filename) as f:
        reader = csv.DictReader(f)
        iter = pool.imap_unordered(make_impression, reader)
        for imp_args in tqdm(iter, total=get_num_lines(filename)):
            imp = Impression(*imp_args)
            train.append(imp)
            if len(imp.features) > num_ones:
                num_ones = len(imp.features)

    print 'building feature map...'
    make_feature_map()

    print 'shuffling training data...'
    np.random.shuffle(train)

    print 'making train and val data...'
    with gzip.open(os.path.join(args.output_dir, 'train.txt.gz'), 'w') as f_train, \
         gzip.open(os.path.join(args.output_dir, 'val.txt.gz'), 'w') as f_val:
        for imp in tqdm(train):
            if imp.clickTime[:2] == '30': # Day 30 as validation set
                write_line(f_val, imp)
            else:
                write_line(f_train, imp)
    
    print 'making test data...'
    with open(os.path.join(args.input_dir, 'test.csv')) as fin, \
         gzip.open(os.path.join(args.output_dir, 'test.txt.gz'), 'w') as fout:
        reader = csv.DictReader(fin)
        for row in tqdm(reader, total=get_num_lines(os.path.join(args.input_dir, 'test.csv'))):
            features, label, clickTime = make_impression(row)
            label = row['instanceID'] # little hack to use write_line()
            imp = Impression(features, label, clickTime)
            write_line(fout, imp)
    
    print 'writing metadata...'
    with open(os.path.join(args.output_dir, 'num_ones.txt'), 'w') as f:
        f.write('{:d}\n'.format(num_ones))
    with open(os.path.join(args.output_dir, 'num_features.txt'), 'w') as f:
        f.write('{:d}\n'.format(num_features))
    with open(os.path.join(args.output_dir, 'feature_map.txt'), 'w') as f:
        for feature, index in feature_map.iteritems():
            f.write('{:d} {:s}\n'.format(index, feature))
    
    print 'all done. waiting python GC...'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--db', type=str, required=True)
    parser.add_argument('--num_workers', type=int, default=multiprocessing.cpu_count() * 2)
    args = parser.parse_args()
    main(args)
