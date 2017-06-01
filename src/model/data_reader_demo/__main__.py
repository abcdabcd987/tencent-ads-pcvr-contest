import time
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
    tic = time.time()
    data = rep.get_dataset(dataset=dataset,
                           batch_size=512,
                           allow_smaller_final_batch=True)
    for i, (xs, ys) in enumerate(data):
        print('step', i, 'xs.shape', xs.shape, 'ys.shape', ys.shape)
    toc = time.time()
    print('read', dataset, 'in', toc-tic, 'seconds')

main()
