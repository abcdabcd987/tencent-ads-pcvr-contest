import argparse
from src.model import mlib
from src import utils, data
from datetime import datetime

class LinearRegressionCTR(mlib.ModelTemplate):
	pass

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--train', action='store_true')
	parser.add_argument('--model', type=str, help='load model from the given path')
	args, _ = parser.parse_known_args()
	return args

def run_model(args):
	args.config["nameid"] = args.nameid
	print("processing nameid: {}".format(args.nameid))

	model = LinearRegressionCTR(args.data_storage, args.config)
	args.data_storage.load_data()
	print('feature data loaded')

	if args.model:
		model.load(args.model)
	print("training...")
	for _ in range(args.config['epoch']):
		model.train(args.train)
		model.save()
		if args.val is not None:
			model.validate(args.val)
	if args.test is not None:
		print('testing...')
		model.test(args.test)

def main():
	config = utils.read_module_config(__file__, 'model.json')
	data_storage = data.DataStorage(config['features'])
	args = parse_args()

	config["session_time"] = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
	modelarg = argparse.Namespace()
	modelarg.config = config
	modelarg.data_storage = data_storage
	modelarg.model = args.model
	for i in range(1, 5):
		modelarg.nameid = str(i)
		modelarg.train = 'smalltrain{}'.format(i)
		modelarg.val = 'val{}'.format(i)
		modelarg.test = None
		run_model(modelarg)
	modelarg.nameid = 'final'
	modelarg.train = 'train'
	modelarg.val = None
	modelarg.test = 'test'
	run_model(modelarg)

main()
