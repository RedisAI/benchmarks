import time
from warnings import warn


class Reporter:
    def __init__(self):
        self._tlist = {}
        self.__enter__()
        self._count = 0

    def __enter__(self):
        self._tlist['__start'] = time.time()
        return self

    def __exit__(self, type, value, traceback):
        self._tlist['__stop'] = time.time()
        self._write_to_disk()
        self.summary()

    def close(self):
        self._tlist['__stop'] = time.time()

    def _step(self, desc):
        if desc in self._tlist.keys():
            raise ValueError(f'Entry for {desc} already saved')
        self._tlist[desc] = time.time()

    def ignore(self, status):
        pass

    @classmethod
    def _write_to_disk(cls):
        pass

    def run(self, count, runner, *args, **kwargs):
        if self._count > 0:
            raise RuntimeError('    run() has been called already')
        else:
            self._count = count
        # dummy run
        runner(*args, **kwargs)
        self._step(f'__count_init')
        for i in range(count):
            runner(*args, **kwargs)
            self._step(f'__count_{i}')

    def summary(self):
        try:
            start = self._tlist.pop('__start')
        except KeyError:
            warn('    Reporter is not initiated properly')
        try:
            stop = self._tlist.pop('__stop')
        except KeyError:
            warn('    Reporter is not exited properly')
        print(f'    Total Time Taken: {stop - start}')
        temp = self._tlist.pop('__count_init')
        foraverage = []
        # dictionaries keeps the order
        for key, val in self._tlist.items():
            foraverage.append(val - temp)
            temp = val
        total = len(foraverage)
        if total != self._count:
            warn('    Something wrong! Raise an issue')
        avg = sum(foraverage) / total
        print(f'    Average time taken for {total} execution: {avg}')

