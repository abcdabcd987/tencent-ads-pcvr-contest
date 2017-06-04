#!/usr/bin/env python3

import argparse
import csv
import sqlite3
import os
import numpy as np
from array import array
from collections import namedtuple, defaultdict
from tqdm import tqdm

from utils import *


FeatureItem = namedtuple('FeatureItem', ['name', 'value'])
def make_raw_item_from_row(row, name):
    return FeatureItem(name, int(row[name]))


def get_installed_app_list(userID, time):
    global cursor
    cursor.execute('''SELECT appID FROM user_installedapps WHERE userID={userID} UNION
        SELECT appID FROM user_app_actions WHERE userID={userID} AND installTime<={installTime}
        '''.format(userID=userID, installTime=time))
    return np.array([x[0] for x in cursor.fetchall()], dtype=np.int32)


def augment_input(row):
    global cursor
    features = []
    
    installedApps = get_installed_app_list(int(row['userID']), int(row['clickTime']))
    features.append(make_raw_item_from_row(row, 'clickTime'))

    cursor.execute('SELECT * FROM ads WHERE creativeID=' + row['creativeID'])
    c = cursor.fetchone()
    features.append(make_raw_item_from_row(c, 'creativeID'))
    features.append(make_raw_item_from_row(c, 'adID'))
    features.append(make_raw_item_from_row(c, 'camgaignID'))
    features.append(make_raw_item_from_row(c, 'advertiserID'))
    features.append(make_raw_item_from_row(c, 'appPlatform'))

    cursor.execute('SELECT * FROM app_categories WHERE appID=' + str(c['appID']))
    a = cursor.fetchone()
    features.append(make_raw_item_from_row(a, 'appID'))
    features.append(make_raw_item_from_row(a, 'appCategory'))

    cursor.execute('SELECT * FROM users WHERE userID=' + row['userID'])
    u = cursor.fetchone()
    features.append(make_raw_item_from_row(u, 'userID'))
    features.append(make_raw_item_from_row(u, 'age'))
    features.append(make_raw_item_from_row(u, 'gender'))
    features.append(make_raw_item_from_row(u, 'education'))
    features.append(make_raw_item_from_row(u, 'marriageStatus'))
    features.append(make_raw_item_from_row(u, 'haveBaby'))
    features.append(make_raw_item_from_row(u, 'hometown'))
    features.append(make_raw_item_from_row(u, 'residence'))

    cursor.execute('SELECT * FROM positions WHERE positionID=' + row['positionID'])
    p = cursor.fetchone()
    features.append(make_raw_item_from_row(p, 'positionID'))
    features.append(make_raw_item_from_row(p, 'sitesetID'))
    features.append(make_raw_item_from_row(p, 'positionType'))

    features.append(make_raw_item_from_row(row, 'connectionType'))
    features.append(make_raw_item_from_row(row, 'telecomsOperator'))

    return features, installedApps


def make_features(filename, total_lines, label_column):
    global args, features, labels, installedApps_list
    with open(os.path.join(args.input_dir, filename)) as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, total=total_lines):
            feature_items, installedApps = augment_input(row)
            label = int(row[label_column])  # label for train, instanceID for test
            labels.append(label)
            for item in feature_items:
                features[item.name].append(item.value)
            installedApps_list.append(installedApps)


def main(args):
    global cursor, num_trains, num_tests, features, labels, installedApps_list

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    num_trains = get_num_lines(os.path.join(args.input_dir, 'train.csv')) - 1
    num_tests = get_num_lines(os.path.join(args.input_dir, 'test.csv')) - 1
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    features = defaultdict(lambda: array('l'))
    installedApps_list = []
    labels = array('l')

    print('building raw features from train...')
    make_features('train.csv', num_trains, 'label')
    
    print('building raw features from test...')
    make_features('test.csv', num_tests, 'instanceID')
    
    print('writing raw features to disk...')
    for feature_name, feature_list in features.items():
        feature_list = np.asarray(feature_list, dtype=np.int32)
        dump_feature(os.path.join(args.output_dir, 'raw', feature_name + '.npy'), feature_list)
    dump_feature(os.path.join(args.output_dir, 'raw', 'installedApps.npy'), installedApps_list)
    labels = np.asarray(labels, dtype=np.int32)
    dump_feature(os.path.join(args.output_dir, 'label.npy'), labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--db', type=str, required=True)
    args = parser.parse_args()
    main(args)
