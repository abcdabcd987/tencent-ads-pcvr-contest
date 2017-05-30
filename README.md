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

Copy the extracted features from NAS to your working directory:

```bash
mkdir -p out/data/features/
cp -r /NAS/Workspaces/tencent-ads-pcvr-contest/features/* out/data/features/
```

## Models

#### DataReader Demo

```bash
python -m src.model.data_reader_demo
```

#### Logistic Regression Demo

```bash
python -m src.model.data_reader_lr_demo
```
