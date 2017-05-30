# tencent-ads-pcvr-contest

## Data Preprocess

```bash
sudo apt-get install -y sqlite
pip install tqdm --user

# convert csv to database for quick random access
./src/data/create-db.sh \
    out/data/raw \
    out/data/preprocess/pre.db

# 3GB Memory, 20 minutes on SSD
./src/data/make-feature-raw.py \
    --input_dir out/data/raw \
    --output_dir out/data/features \
    --db out/data/preprocess/pre.db

# 1GB Memory, 3 minutes
python -m src.data.make_feature_basic

# 6GB Memory, 6 minutes
python -m src.data.make_feature_installedApps

# other features
python -m src.data.make_feature ACTION
```

## Models

#### DataReader Demo

```bash
python -m src.model.data_reader_demo
```

#### (Deprecated) `model_20170504_clq_naive_lr`

```bash
export data_root=out/pre-20170504-naive
python model/model_20170504_clq_naive_lr.py \
    --data_root $data_root/ \
    --output_root out/ \
    --num_feature $(cat $data_root/num_features.txt) \
    --num_one $(cat $data_root/num_ones.txt)
tensorboard --logdir=out/logs --reload_interval 2 --port 6006
```
