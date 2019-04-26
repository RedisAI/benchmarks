import time
from warnings import warn
from collections import defaultdict

from tqdm import tqdm
import seaborn as sns


sns.set(style='whitegrid')


def save_chart(title, x_label, y_label, x_data, y_data):
    if len(x_data) != len(y_data):
        raise Exception('Length of X is not same as Y')
    bplot = sns.barplot(x_data, y_data)
    bplot.set(xlabel=x_label, ylabel=y_label)
    bplot.set_title(title)
    fig = bplot.get_figure()
    print(f'Saving {title}.png ..')
    filename = title.replace(' ', '_')
    fig.savefig(f'{filename}.png')


class Reporter:

    accumulate: dict = defaultdict(dict)

    def __init__(self, backend, model, experiment, device):
        self._tlist = {}
        self.__enter__()
        self._count = 0
        self._title = f'{model} running on {backend}:{device}'
        self._experiment = experiment

    def __enter__(self):
        self._tlist['__start'] = time.time()
        return self

    def __exit__(self, type, value, traceback):
        self._tlist['__stop'] = time.time()
        self.accumulate[self._title][self._experiment] = self._get_average()

    def close(self):
        self._tlist['__stop'] = time.time()

    def _step(self, desc):
        if desc in self._tlist.keys():
            raise ValueError(f'Entry for {desc} already saved')
        self._tlist[desc] = time.time()

    def run(self, count, runner, *args, **kwargs):
        if self._count > 0:
            raise RuntimeError('    run() has been called already')
        else:
            self._count = count
        # dummy run
        runner(*args, **kwargs)
        self._step(f'__count_init')
        for i in tqdm(range(count)):
            yield runner(*args, **kwargs)
            self._step(f'__count_{i}')

    def _get_average(self):
        try:
            start = self._tlist.pop('__start')
        except KeyError:
            warn('    Reporter is not initiated properly')
        try:
            stop = self._tlist.pop('__stop')
        except KeyError:
            warn('    Reporter is not exited properly')

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
        return avg

    @classmethod
    def summarize(cls):
        for title, data in cls.accumulate.items():
            save_chart(
                title,
                'Experiments', 'Average time taken',
                list(data.keys()), list(data.values()))
