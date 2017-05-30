import mlib

class LinearRegressionCTR(mlib.ModelTemplate):
	def __init__(self, **kwargs):
		self._num_feature = kwargs.pop("num_feature")
		super(self.__class__, self).__init__(self, kwargs)

	def _body(self):
		w = tf.get_variable('weight', [self._num_feature], dtype=tf.float32,
							initializer=tf.random_uniform_initializer(-0.05, 0.05))
		b = tf.get_variable('bias', [1], dtype=tf.float32,
							initializer=tf.zeros_initializer())
		wx = tf.reduce_sum(tf.gather(w, self._ph_x), axis=1)
		return wx + b

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--output_root', type=str, required=True)
    parser.add_argument('--num_feature', type=int, required=True)
    parser.add_argument('--num_one', type=int, required=True)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--model', type=str, help='load model from the given path')
    args, _ = parser.parse_known_args()

    data_root = args.data_root
    output_root = args.output_root
    model = LinearRegressionCTR(data_root=data_root,
                                output_root=output_root,
                                num_feature=args.num_feature,
                                num_one=args.num_one,
                                learning_rate=5e-4,
                                batch_size=512)
    if args.model:
        model.load(args.model)
    if args.train:
        print('training...')
        for _ in range(10):
            model.train()
            model.save()
            model.validate()
    model.test()

if __name__ == '__main__':
    main()
