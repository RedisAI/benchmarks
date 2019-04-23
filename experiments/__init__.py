from core.benchmarker import BenchmarkManager
from . import _pytorch
from . import _tensorflow


def register_experiments():
    BenchmarkManager.register('pytorch.resnet.native', _pytorch.native_exp.run)
    BenchmarkManager.register('pytorch.resnet.redisai', _pytorch.redisai_exp.run)
    BenchmarkManager.register('tensorflow.resnet.native', _tensorflow.native_exp.run)
    BenchmarkManager.register('tensorflow.resnet.redisai', _tensorflow.redisai_exp.run)
    BenchmarkManager.register('tensorflow.resnet.tfs', _tensorflow.tfs_exp.run)
