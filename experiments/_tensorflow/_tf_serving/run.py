import grpc
import numpy
import tensorflow as tf
from skimage import io

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import numpy as np


channel = grpc.insecure_channel('localhost:8500')
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
request = predict_pb2.PredictRequest()
request.model_spec.name = 'resnet'
request.model_spec.signature_name = 'serving_default'


filepath = '/home/hhsecond/mypro/benchmarks/experiments/yolov3/TFS/guitar.jpg'
numpy_img = io.imread(filepath).astype(dtype=np.float32)
numpy_img = np.expand_dims(numpy_img, axis=0) / 255

request.inputs['images'].CopyFrom(
    tf.contrib.util.make_tensor_proto(numpy_img))
result_future = stub.Predict.future(request, 10.25)
res = result_future.result()
response = numpy.array(res.outputs['output'].float_val)
argmx = numpy.argmax(response)

# docker run -p 8500:8500 -p 8501:8501 --name tfserving --mount type=bind,source=/home/hhsecond/mypro/benchmarks/experiments/ServingTF/build,target=/models/resnet -e MODEL_NAME=resnet -t tensorflow/serving
# response = numpy.array(res.outputs['dense_2/Softmax:0'].float_val)
# prediction = numpy.argmax(response)
# print(prediction, response[prediction])
