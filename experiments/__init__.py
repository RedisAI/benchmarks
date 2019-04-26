from . import _pytorch
from . import _tensorflow


def register_experiments(BenchmarkManager):
    BenchmarkManager.register('pytorch.resnet.native', _pytorch.native_client.run)
    BenchmarkManager.register('pytorch.resnet.redisai', _pytorch.redisai_client.run)
    BenchmarkManager.register('pytorch.resnet.flask', _pytorch.flask_client.run)
    BenchmarkManager.register('pytorch.resnet.grpc', _pytorch.grpc_client.run)
    BenchmarkManager.register('tensorflow.resnet.native', _tensorflow.native_client.run)
    BenchmarkManager.register('tensorflow.resnet.flask', _tensorflow.flask_client.run)
    BenchmarkManager.register('tensorflow.resnet.redisai', _tensorflow.redisai_client.run)
    BenchmarkManager.register('tensorflow.resnet.grpc', _tensorflow.tfs_client.run)
