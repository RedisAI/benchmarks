import grpc
from .protofiles.imagedata_pb2 import ImageData
from .protofiles.imagedata_pb2_grpc import PredictorStub
from experiments.utils import get_one_image


def init(config):
    img = get_one_image(transpose=(2, 0, 1))
    imgdata = ImageData()
    # protobuf assumes the shape of the image is (1, 3, height, width)
    # where 1 is the batchsize and 3 is number of channels
    imgdata.image = img.tobytes()
    imgdata.height = img.shape[2]
    imgdata.width = img.shape[3]
    imgdata.dtype = img.dtype.name
    init.image = imgdata
    channel = grpc.insecure_channel('localhost:50051')
    init.stub = PredictorStub(channel)


def run(config, reporter):
    print('run')
    init(config)
    with reporter:
        generator = reporter.run(config['exp_count'], init.stub.GetPrediction, init.image)
        for output in generator:
            assert len(output.output) == 1000
