import redisai as rai
from redisai import model as raimodel

from core.reporter import Reporter
from experiments.utils import get_one_image
from core.config import ConfigManager


con = rai.Client()
graph = raimodel.Model.load(ConfigManager.tf_model_path)

inputs = ['images']
outputs = ['output']

con.modelset(
    'graph', rai.Backend.tf, rai.Device.cpu, graph,
    input=inputs, output=outputs)

image = get_one_image()
image = rai.BlobTensor.from_numpy(image)


def wrapper():
    con.tensorset('image', image)
    con.modelrun('graph', input=['image'], output=['output'])


def run():
    with Reporter() as reporter:
        reporter.run(ConfigManager.exp_count, wrapper)
