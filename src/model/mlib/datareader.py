import sys, gzip, multiprocessing, numpy as np

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