# tencent-ads-pcvr-contest

## Data Preprocess

```bash
sudo apt-get install -y sqlite
pip install tqdm --user

./src/data/create-db.sh \
    out/data/raw \
    out/data/preprocess/pre.db

# 6.5GB Memory, 20 minutes on SSD
./src/data/make-feature-raw.py \
    --input_dir out/data/raw \
    --output_dir out/data/features \
    --db out/data/preprocess/pre.db

# 2GB Memory
./src/data/make-feature-basic.py --feature_dir out/data/features

# 40GB Memory, 10 minutes
./src/data/make-feature-installedApps.py --feature_dir out/data/features

# other features
./src/data/make-feature.py ACTION --feature_dir out/data/features
```

## Models

#### `model_20170504_clq_naive_lr`

```bash
export data_root=out/pre-20170504-naive
python model/model_20170504_clq_naive_lr.py \
    --data_root $data_root/ \
    --output_root out/ \
    --num_feature $(cat $data_root/num_features.txt) \
    --num_one $(cat $data_root/num_ones.txt)
tensorboard --logdir=out/logs --reload_interval 2 --port 6006
```
