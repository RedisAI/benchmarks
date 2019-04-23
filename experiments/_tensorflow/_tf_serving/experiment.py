import grpc
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from core.reporter import Reporter
from experiments.utils import get_one_image
from core.config import ConfigManager

channel = grpc.insecure_channel('localhost:8500')
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
request = predict_pb2.PredictRequest()
request.model_spec.name = 'resnet'
request.model_spec.signature_name = 'serving_default'


image = get_one_image()


def wrapper():
    request.inputs['images'].CopyFrom(tf.contrib.util.make_tensor_proto(image))
    # TODO: what the heck is this 10.25
    result_future = stub.Predict(request, 10.25)
    # result_future.add_done_callback(_callback)


def run():
    with Reporter() as reporter:
        reporter.run(ConfigManager.exp_count, wrapper)


"""
docker run -p 8500:8500 -p 8501:8501 --name tfserving \
--mount type=bind,source=/home/hhsecond/mypro/benchmarks/experiments/data/tf_serving_builds/,\
target=/models/resnet -e MODEL_NAME=resnet -t tensorflow/serving
"""
