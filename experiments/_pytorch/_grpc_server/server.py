from concurrent import futures
import grpc
import time
import torch
import numpy as np

from protofiles.imagedata_pb2_grpc import PredictorServicer
from protofiles.imagedata_pb2_grpc import add_PredictorServicer_to_server
from protofiles.imagedata_pb2 import PredictionClass


pt_model_path = '/root/data/resnet50.pt'
model = torch.jit.load(pt_model_path)


class Predictor(PredictorServicer):

    def GetPrediction(self, request, context):
        shape = (1, 3, request.height, request.width)
        dtype = request.dtype
        image = np.frombuffer(request.image, dtype=dtype).reshape(shape)
        with torch.no_grad():
            out = model(torch.from_numpy(image))
        return PredictionClass(output=out[0].numpy().tolist())


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    add_PredictorServicer_to_server(Predictor(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    try:
        while True:
            time.sleep(10000)
    except KeyboardInterrupt:
        print('Stopping server ...')
        server.stop(0)


if __name__ == '__main__':
    serve()
