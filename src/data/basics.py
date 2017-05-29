#!/usr/bin/env python2

import argparse
import csv
import sqlite3
import os
import json
import cPickle
from collections import namedtuple
from tqdm import tqdm


FeatureItem = namedtuple('FeatureItem', ['name', 'value'])
Impression = namedtuple('Impression', ['label', 'clickTime', 'features'])


def make_onehot_from_row(row, name):
    return FeatureItem(name, row[name])


def make_impression(row):
    global cursor
    label = int(row['label'])
    clickTime = int(row['clickTime'])
    features = []

    cursor.execute('SELECT * FROM ads WHERE creativeID=' + row['creativeID'])
    c = cursor.fetchone()
    features.append(make_onehot_from_row(c, 'creativeID'))
    features.append(make_onehot_from_row(c, 'adID'))
    features.append(make_onehot_from_row(c, 'camgaignID'))
    features.append(make_onehot_from_row(c, 'advertiserID'))
    features.append(make_onehot_from_row(c, 'appPlatform'))

    cursor.execute('SELECT * FROM app_categories WHERE appID=' + str(c['appID']))
    a = cursor.fetchone()
    features.append(make_onehot_from_row(a, 'appID'))
    features.append(make_onehot_from_row(a, 'appCategory'))

    cursor.execute('SELECT * FROM users WHERE userID=' + row['userID'])
    u = cursor.fetchone()
    features.append(make_onehot_from_row(u, 'userID'))
    features.append(make_onehot_from_row(u, 'age'))
    features.append(make_onehot_from_row(u, 'gender'))
    features.append(make_onehot_from_row(u, 'education'))
    features.append(make_onehot_from_row(u, 'marriageStatus'))
    features.append(make_onehot_from_row(u, 'haveBaby'))
    features.append(make_onehot_from_row(u, 'hometown'))
    features.append(make_onehot_from_row(u, 'residence'))

    cursor.execute('SELECT * FROM positions WHERE positionID=' + row['positionID'])
    p = cursor.fetchone()
    features.append(make_onehot_from_row(p, 'positionID'))
    features.append(make_onehot_from_row(p, 'sitesetID'))
    features.append(make_onehot_from_row(p, 'positionType'))

    features.append(make_onehot_from_row(row, 'connectionType'))
    features.append(make_onehot_from_row(row, 'telecomsOperator'))

    return label, clickTime, features


def make_feature_maps():
    global args, feature_maps, feature_counts, num_trains
    feature_counts = {}
    with open(os.path.join(args.input_dir, 'train.csv')) as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, total=num_trains):
            label, clickTime, features = make_impression(row)
            for item in features:
                count = feature_counts.get(item.name, None)
                if count is None:
                    count = {}
                    feature_counts[item.name] = count
                count[item.value] = count.get(item.value, 0) + 1
    
    feature_maps = {}
    for feature_name, category_count in feature_counts.iteritems():
        l = sorted(category_count.iteritems(), key=lambda (category, count): category)
        m = {'__other__': 0}
        feature_maps[feature_name] = m
        for category, count in l:
            if count >= 4:
                m[category] = len(m)


def make_features(filename, total_lines, label_column):
    global args, features, labels, clickTimes
    with open(os.path.join(args.input_dir, filename)) as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, total=num_trains):
            label, clickTime, feature_items = make_impression(row)
            label = int(row[label_column])  # label for train, instanceID for test
            labels.append(label)
            clickTimes.append(clickTime)
            for item in feature_items:
                feature_map = feature_maps[item.name]
                idx = feature_map.get(item.value, None)
                if idx is None:
                    idx = feature_map['__other__']
                features[item.name].append(idx)


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


def main(args):
    global feature_maps, feature_counts, cursor, num_trains, num_tests, features, labels, clickTimes

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    num_trains = get_num_lines(os.path.join(args.input_dir, 'train.csv')) - 1
    num_tests = get_num_lines(os.path.join(args.input_dir, 'test.csv')) - 1
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    print 'building feature maps...'
    make_feature_maps()

    features = {name: [] for name in feature_maps}
    labels = []
    clickTimes = []

    print 'building features from train...'
    make_features('train.csv', num_trains, 'label')
    
    print 'building features from test...'
    make_features('test.csv', num_tests, 'instanceID')
    
    print 'writing features to disk...'
    for feature_name in features:
        m = feature_maps[feature_name]
        c = feature_counts[feature_name]
        dump_meta(os.path.join(args.output_dir, 'basic', feature_name + '.meta.json'),
                  {'type': 'one_hot', 'dimension': len(m), 'index': m, 'count': c})
        dump_feature(os.path.join(args.output_dir, 'basic', feature_name + '.pkl'), features[feature_name])
    dump_meta(os.path.join(args.output_dir, 'basic', 'clickTime.meta.json'),
              {'type': 'numeric', 'dimension': 1})
    dump_feature(os.path.join(args.output_dir, 'basic', 'clickTime.pkl'), clickTimes)
    dump_feature(os.path.join(args.output_dir, 'label.pkl'), labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--db', type=str, required=True)
    args = parser.parse_args()
    main(args)
