import argparse
from .. import mlib

class LinearRegressionCTR(mlib.ModelTemplate):
	pass

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--train', action='store_true')
	parser.add_argument('--model', type=str, help='load model from the given path')
	args, _ = parser.parse_known_args()
	return args

def run_session(args, config):
	sess = mlib.Session(config)
	if args.model:
		sess.load(args.model)
	print("training...")
	for _ in range(config.epoch):
		sess.train(config.train)
		if config.val is not None:
			sess.validate(config.val)
	if config.test is not None:
		print('testing...')
		sess.test(config.test)
	sess.save()

def main():
	args = parse_args()
	config = mlib.Config.from_json(__file__, 'model.json')
	mlib.Datasets(config.data_config)
	LinearRegressionCTR(config.model_config)

	for i in [1, 2, 3, 4]:
		run_session(args, config.session_config(i))
	run_session(args, config.session_config('final'))

main()
