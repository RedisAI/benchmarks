import redisai as rai
from redisai import model as raimodel

from experiments.utils import get_one_image


def init(config):
    model = raimodel.Model.load(config['modelpath'])
    host = config['server'].split(':')[0]
    port = config['server'].split(':')[1]
    init.con = rai.Client(host=host, port=port)
    init.con.modelset('model', rai.Backend.torch, rai.Device.cpu, model)
    image, init.img_class = get_one_image(transpose=(2, 0, 1))
    init.image = rai.BlobTensor.from_numpy(image)


def wrapper(init):
    init.con.tensorset('image', init.image)
    init.con.modelrun('model', input=['image'], output=['out'])
    return init.con.tensorget('out', as_type=rai.BlobTensor).to_numpy()


def run(config, reporter):
    init(config)
    with reporter:
        generator = reporter.run(config['exp_count'], wrapper, init)
        for output in generator:
            assert output.shape == (1, 1000)
            assert output.argmax() == init.img_class
