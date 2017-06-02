import time
import numpy as np
from ...data import *
from ...utils import *

def main():
    model_config = read_module_config(__file__, 'model.json')
    print('model configuration is loaded')

    data_storage = DataStorage(model_config['features'])
    print('feature metadata is loaded')

    rep = data_storage.get_representation(IndexRepresentation)
    print('dense_shape', rep.dense_shape)
    print('max_length', rep.max_length)

    data_storage.load_data()
    print('feature data is loaded')

    dataset = 'val1'
    min_rowid = float('+inf')
    max_rowid = float('-inf')
    tic = time.time()
    data = rep.get_dataset(dataset=dataset,
                           batch_size=512,
                           allow_smaller_final_batch=True)
    for i, (xs, ys, rowids) in enumerate(data):
        print('step', i, 'xs.shape', xs.shape, 'ys.shape', ys.shape, 'rowids.shape', rowids.shape)
        min_rowid = min(min_rowid, np.min(rowids))
        max_rowid = max(max_rowid, np.max(rowids))
    toc = time.time()
    print('min_rowid', min_rowid, 'max_rowid', max_rowid)
    print('read', dataset, 'in', toc-tic, 'seconds')

main()
