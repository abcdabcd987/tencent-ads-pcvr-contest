# tencent-ads-pcvr-contest

[Google Docs](https://docs.google.com/document/d/1WjTEfZee6BpXMGkQAVd-5T3vTD2y9mkwSt-w0djw0WA/edit?usp=sharing)

This project uses Python 3.

## Data Preprocess

```bash
sudo apt-get install -y sqlite
pip3 install tqdm --user

# convert csv to database for quick random access
./src/data/create-db.sh \
    out/data/raw \
    out/data/preprocess/pre.db

# 3GB memory, 20 minutes on SSD
./src/data/make-feature-raw.py \
    --input_dir out/data/raw \
    --output_dir out/data/features \
    --db out/data/preprocess/pre.db

# 1GB memory, 3 minutes
python3 -m src.data.make_feature_basic

# 6GB memory, 6 minutes
python3 -m src.data.make_feature_installedApps

# other features
python3 -m src.data.make_feature ACTION
```

Copy the extracted features from NAS to your working directory:

```bash
mkdir -p out/data/features/
cp -r /NAS/Workspaces/tencent-ads-pcvr-contest/features/* out/data/features/
```

## Models

#### DataReader Demo

```bash
# 1GB memory
python3 -m src.model.data_reader_demo
```

#### Logistic Regression Demo

```bash
# 4GB memory
python3 -m src.model.data_reader_lr_demo --train --test --predict
```
