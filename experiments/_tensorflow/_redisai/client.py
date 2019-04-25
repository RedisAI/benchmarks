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
    image = get_one_image()
    init.image = rai.BlobTensor.from_numpy(image)


def wrapper(init):
    init.con.tensorset('image', init.image)
    init.con.modelrun('graph', input=['image'], output=['output'])


def run(config, Reporter):
    init(config)
    with Reporter() as reporter:
        reporter.run(config['exp_count'], wrapper, init)
