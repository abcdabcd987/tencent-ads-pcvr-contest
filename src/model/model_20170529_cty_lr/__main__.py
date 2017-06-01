from src.model import mlib
from src import utils, data

class LinearRegressionCTR(mlib.ModelTemplate):
    pass

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--model', type=str, help='load model from the given path')
    args, _ = parser.parse_known_args()
    return args

def main():
    config = utils.read_module_config(__file__, 'model.json')
    data_storage = data.DataStorage(config['features'])
    args = parse_args()

    model = LogisticRegressionCTR(data_storage)
    data_storage.load_data()
    print('feature data loaded')
    if args.model:
        model.load(args.model)
    if args.train:
        print('training...')
        for _ in range(config['epoch']):
            model.train('smalltrain1')
            model.save()
            model.validate('val1')
    print('testing...')
    model.test('test')

main()
