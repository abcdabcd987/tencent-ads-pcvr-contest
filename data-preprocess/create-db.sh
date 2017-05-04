#!/bin/bash

INPUT_DIR=$1
OUTPUT_DB=$2

if [ "$#" -ne 2 ]; then
    echo "usage: ./create-db.sh <input-dir> <output-db>"
    exit 1
fi

if [ -e "$OUTPUT_DB" ]; then
    echo $OUTPUT_DB exists. abort.
    exit 1
fi

if [ ! -d "$INPUT_DIR" ]; then
    echo $INPUT_DIR does not exist. abort.
    exit 1
fi

cat << SQL | sqlite3 $OUTPUT_DB
CREATE TABLE users (
    userID INTEGER,
    age INTEGER,
    gender INTEGER,
    education INTEGER,
    marriageStatus INTEGER,
    haveBaby INTEGER,
    hometown INTEGER,
    residence INTEGER
);
.separator ","
.headers on
.import $INPUT_DIR/user.csv users
CREATE INDEX uesrs_userID_idx ON users (userID);
SQL
echo done: users

cat << SQL | sqlite3 $OUTPUT_DB
CREATE TABLE user_installedapps (
    userID INTEGER,
    appID INTEGER
);
.separator ","
.headers on
.import $INPUT_DIR/user_installedapps.csv user_installedapps
CREATE INDEX user_installedapps_userID_idx ON user_installedapps (userID);
SQL
echo done: user_installedapps

cat << SQL | sqlite3 $OUTPUT_DB
CREATE TABLE user_app_actions (
    userID INTEGER,
    installTime INTEGER,
    appID INTEGER
);
.separator ","
.headers on
.import $INPUT_DIR/user_app_actions.csv user_app_actions
CREATE INDEX user_app_actions_userID_idx ON user_app_actions (userID);
CREATE INDEX user_app_actions_userID_installTime_idx ON user_app_actions (userID, installTime);
SQL
echo done: user_app_actions

cat << SQL | sqlite3 $OUTPUT_DB
CREATE TABLE app_categories (
    appID INTEGER,
    appCategory INTEGER
);
.separator ","
.headers on
.import $INPUT_DIR/app_categories.csv app_categories
CREATE INDEX app_categories_appID_idx ON app_categories (appID);
SQL
echo done: app_categories

cat << SQL | sqlite3 $OUTPUT_DB
CREATE TABLE ads (
    creativeID INTEGER,
    adID INTEGER,
    camgaignID INTEGER,
    advertiserID INTEGER,
    appID INTEGER,
    appPlatform INTEGER
);
.separator ","
.headers on
.import $INPUT_DIR/ad.csv ads
CREATE INDEX ads_creativeID_idx ON ads (creativeID);
SQL
echo done: ads

cat << SQL | sqlite3 $OUTPUT_DB
CREATE TABLE positions (
    positionID INTEGER,
    sitesetID INTEGER,
    positionType INTEGER
);
.separator ","
.headers on
.import $INPUT_DIR/position.csv positions
CREATE INDEX positions_positionID_idx ON positions (positionID);
SQL
echo done: positions

echo all done
