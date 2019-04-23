import grpc
import numpy
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import numpy as np
from keras.datasets import mnist
import time

tf.app.flags.DEFINE_integer('concurrency', 1,
                            'maximum number of concurrent inference requests')
tf.app.flags.DEFINE_integer('num_tests', 100, 'Number of test images')
tf.app.flags.DEFINE_string('server', '', 'PredictionService host:port')
tf.app.flags.DEFINE_string('work_dir', '/tmp', 'Working directory. ')
FLAGS = tf.app.flags.FLAGS

_counter = 0
_start = 0


def _callback(result_future):
    """Callback function.
    Calculates the statistics for the prediction result.
    Args:
      result_future: Result future of the RPC.
    """
    global _counter
    global _start
    exception = result_future.exception()
    if exception:
        print(exception)
    else:
        response = numpy.array(
            result_future.result().outputs['dense_2/Softmax:0'].float_val)
        prediction = numpy.argmax(response)
        _counter += 1
        if((_counter % 100) == 0):
            print(_counter, prediction, response[prediction])


def do_inference(hostport, work_dir, concurrency, num_tests):
    """Tests PredictionService with concurrent requests.
    Args:
        hostport: Host:port address of the PredictionService.
        work_dir: The full path of working directory for test data set.
        concurrency: Maximum number of concurrent requests.
        num_tests: Number of test images to use.
    Returns:
        The classification error rate.
    Raises:
        IOError: An error occurred processing test data set.
    """
    channel = grpc.insecure_channel(hostport)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'mnist'
    request.model_spec.signature_name = 'serving_default'

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)

    x = X_train[4545][0]
    print(x.shape, y_train[4545])

    for _ in range(num_tests):
        x = x.astype(np.float32)
        request.inputs['input_image'].CopyFrom(
            tf.contrib.util.make_tensor_proto(x, shape=[1, 1, 28, 28]))
        result_future = stub.Predict.future(request, 10.25)
        result_future.add_done_callback(_callback)

    res = result_future.result()
    response = numpy.array(res.outputs['dense_2/Softmax:0'].float_val)
    prediction = numpy.argmax(response)
    print(prediction, response[prediction])


def main(_):
  if FLAGS.num_tests > 20000:
    print('num_tests should not be greater than 20k')
    return
  if not FLAGS.server:
    print('please specify server host:port')
    return
  error_rate = do_inference(FLAGS.server, FLAGS.work_dir,
                            FLAGS.concurrency, FLAGS.num_tests)  

if __name__ == '__main__':
    print ("hello from TFServing client slim")
    tf.app.run()

