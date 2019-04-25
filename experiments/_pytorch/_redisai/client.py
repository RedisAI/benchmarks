import redisai as rai
from redisai import model as raimodel

from experiments.utils import get_one_image


def init(config):
    model = raimodel.Model.load(config['modelpath'])
    host = config['server'].split(':')[0]
    port = config['server'].split(':')[1]
    init.con = rai.Client(host=host, port=port)
    init.con.modelset('model', rai.Backend.torch, rai.Device.cpu, model)
    image = get_one_image(transpose=(2, 0, 1))
    init.image = rai.BlobTensor.from_numpy(image)


def wrapper(init):
    init.con.tensorset('image', init.image)
    init.con.modelrun('model', input=['image'], output=['out'])


def run(config, Reporter):
    init(config)
    with Reporter() as reporter:
        reporter.run(config['exp_count'], wrapper, init)
