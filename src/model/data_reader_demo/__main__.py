from ...data import *
from ...utils import *

def main():
    model_config = read_module_config(__file__, 'model.json')
    data_reader = DataReader(model_config['features'])
    val1 = data_reader.get_dataset('test', IndexBatch, 4)
    print 'dense_shape', val1.dense_shape
    print 'max_length', val1.max_length
    data_reader.load_data()
    for xs, ys in val1:
        print 'xs', xs
        print 'ys', ys
