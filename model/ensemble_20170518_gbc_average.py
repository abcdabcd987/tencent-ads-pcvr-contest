import argparse
import sys
import os
import gzip
import multiprocessing

import numpy as np
import tensorflow as tf


class EnsembleModel(object):
    def __init__(self, **kargs):
        self.model_root = kargs['model_root']

class BufferedDataReader(object):
    def __init__(self, filename, batch_size, num_one, batch_buffer_size=100, num_batch_worker=4):
        self._line_queue = multiprocessing.Queue(batch_buffer_size * batch_size)
        self._batch_queue = multiprocessing.Queue(batch_buffer_size)
        self._batch_maker = []
        if batch_size == -1:
            batch_size = sys.maxint
        for _ in range(num_batch_worker):
            p = multiprocessing.Process(target=BufferedDataReader._batch_worker,
                                        args=(self._line_queue, self._batch_queue, batch_size, num_one))
            self._batch_maker.append(p)
            p.start()
        self._line_reader = multiprocessing.Process(target=BufferedDataReader._line_worker,
                                                    args=(self._line_queue, filename, len(self._batch_maker)))
        self._line_reader.start()

    def qsize(self):
        return self._line_queue.qsize(), self._batch_queue.qsize()

    def stop(self):
        self._line_reader.terminate()
        for p in self._batch_maker:
            p.terminate()

    def join(self):
        self._line_reader.join()
        for p in self._batch_maker:
            p.join()

    def get_batch(self):
        xs, ys = self._batch_queue.get()
        return xs, ys

    @staticmethod
    def _batch_worker(line_queue, batch_queue, batch_size, num_one):
        run = True
        while run:
            xs = np.zeros((batch_size, num_one), dtype=int)
            ys = np.empty(batch_size, dtype=int)
            for i in range(batch_size):
                line = line_queue.get()
                if line is None:
                    xs, ys = xs[:i], ys[:i]
                    run = False
                    break

                split = line.split()
                clicked = int(split[0])
                for j, s in enumerate(split[1:]):
                    k, v = s.split(':')
                    xs[i, j] = int(k)
                ys[i] = clicked
            batch_queue.put((xs, ys))
            run = run and len(ys) != 0
        batch_queue.put(([], []))
        batch_queue.close()

    @staticmethod
    def _line_worker(line_queue, filename, num_batch_maker):
        with gzip.open(filename) as f:
            for line in f:
                line_queue.put(line)
        for _ in range(num_batch_maker):
            line_queue.put(None)
        line_queue.close()


def test():
    saver = tf.train.import_meta_graph(os.path.join(model_root, 'model-{}.meta'.format(loop)))
    print(os.path.join(model_root, 'model-{}'.format(loop)))
    saver.restore(save_path=os.path.join(model_root, 'model-{}'.format(loop)), sess=sess)
    graph = tf.get_default_graph()
    # for n in graph.as_graph_def().node:
    #     print(n)
    
    ph_x = graph.get_tensor_by_name("ph_x:0")
    logits = graph.get_tensor_by_name("logits:0")
    prob = graph.get_tensor_by_name("prob:0")
    print(logits)
    print(prob)
    res = []
    filename = os.path.join(data_root, 'test.txt.gz')
    reader = BufferedDataReader(filename, 512, num_one)
    try:
        while True:
            xs, ys = reader.get_batch()
            if len(ys) == 0:
                break
            #probs = sess.run(tf.sigmoid(logits), feed_dict={ph_x: xs})
            probs = sess.run(prob, feed_dict={ph_x: xs})
            print("yes!")
            for i, prob in zip(ys, probs):
                res.append((i, prob))
    except:
        reader.stop()
        reader.join()
        raise
    reader.stop()
    reader.join()
    res.sort(key=lambda (i, prob): i)

    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_root', type=str, required=True)
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--model_num', type=int, required=True)
    parser.add_argument('--num_feature', type=int, required=True)
    parser.add_argument('--num_one', type=int, required=True)
    args, _ = parser.parse_known_args()

    model_root = args.model_root
    data_root = args.data_root
    model_num = args.model_num
    num_feature = args.num_feature
    num_one = args.num_one
    # ensemble_model = EnsembleModel(data_root=data_root,
    #                                model_root=model_root)

    #build the model
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # ph_x = tf.placeholder(tf.int32, [None, num_one])
    # ph_y = tf.placeholder(tf.float32, [None])
    # w = tf.get_variable('weight', [num_feature], dtype=tf.float32,
    #                     initializer=tf.random_uniform_initializer(-0.05, 0.05))
    # b = tf.get_variable('bias', [1], dtype=tf.float32,
    #                     initializer=tf.zeros_initializer())
    # wx = tf.reduce_sum(tf.gather(w, ph_x), axis=1)
    # logits = wx + b
    # prob = tf.sigmoid(logits)



    preds = []
    for loop in range(model_num):
        print(loop)
        pred = test()
        preds.append(pred)

    print(preds)

