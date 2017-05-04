# tencent-ads-pcvr-contest

## Data Preprocess

```bash
sudo apt-get install -y sqlite
pip install tqdm --user
./data-preprocess/create-db.sh INPUT_DIR OUTPUT_DB
./data-preprocess/makedata.py --input_dir INPUT_DIR --output_dir OUTPUT_DIR --db DB
```
