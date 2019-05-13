import redisai as rai
from redisai import model as raimodel

from experiments.utils import get_one_image


def init(config):
    host = config['server'].split(':')[0]
    port = config['server'].split(':')[1]
    init.con = rai.Client(host=host, port=port)
    graph = raimodel.Model.load(config['modelpath'])
    inputs = ['images']
    outputs = ['output']
    init.con.modelset(
        'graph', rai.Backend.tf, rai.Device.cpu, graph,
        input=inputs, output=outputs)
    image, init.img_class = get_one_image()
    init.image = rai.BlobTensor.from_numpy(image)


def wrapper(init):
    init.con.tensorset('image', init.image)
    init.con.modelrun('graph', input=['image'], output=['output'])
    return init.con.tensorget('output', as_type=rai.BlobTensor).to_numpy()


def run(config, reporter):
    init(config)
    with reporter:
        generator = reporter.run(config['exp_count'], wrapper, init)
        for output in generator:
            assert output.shape == (1, 1001)
            assert output.argmax() - 1 == init.img_class
