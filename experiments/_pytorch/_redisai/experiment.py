import redisai as rai
from redisai import model as raimodel

from core.reporter import Reporter
from experiments.utils import get_one_image
from core.config import ConfigManager


model = raimodel.Model.load(ConfigManager.pt_model_path)
con = rai.Client()
con.modelset('model', rai.Backend.torch, rai.Device.cpu, model)
transpose = (2, 0, 1)
image = get_one_image(transpose=transpose)
image = rai.BlobTensor.from_numpy(image)


def wrapper():
    con.tensorset('image', image)
    con.modelrun('model', input=['image'], output=['out'])


def run():
    with Reporter() as reporter:
        reporter.run(ConfigManager.exp_count, wrapper)
