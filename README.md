# tencent-ads-pcvr-contest

## Data Preprocess

Need at least 28GB memory. Should be done in about 30 minutes on HDD.

```bash
sudo apt-get install -y sqlite
pip install tqdm --user
./data-preprocess/create-db.sh INPUT_DIR OUTPUT_DB
./data-preprocess/makedata.py --input_dir INPUT_DIR --output_dir OUTPUT_DIR --db DB
```

## Models

#### `model_20170504_clq_naive_lr`

```bash
export data_root=out/pre-20170504-naive
python model/model_20170504_clq_naive_lr.py \
    --data_root $data_root/ \
    --output_root out/ \
    --num_feature $(cat $data_root/num_features.txt)
tensorboard --logdir=out/logs --reload_interval 2 --port 6006
```
