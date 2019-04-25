import grpc
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from experiments.utils import get_one_image


def init(config):
    channel = grpc.insecure_channel('localhost:8500')
    init.stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    init.request = predict_pb2.PredictRequest()
    init.request.model_spec.name = 'resnet'
    init.request.model_spec.signature_name = 'serving_default'
    init.image = get_one_image()


def wrapper(init):
    init.request.inputs['images'].CopyFrom(
        tf.contrib.util.make_tensor_proto(init.image))
    result_future = init.stub.Predict(init.request, 10.25)
    return result_future
    # TODO: what the heck is this 10.25
    # TODO: result_future.add_done_callback(_callback)
    # TODO: make sure wrapper is doing new request


def run(config, Reporter):
    init(config)
    with Reporter() as reporter:
        reporter.run(config['exp_count'], wrapper, init)
