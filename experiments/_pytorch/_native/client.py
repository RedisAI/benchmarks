import torch
from experiments.utils import get_one_image


def init(config):
    init.model = torch.jit.load(config['modelpath'])
    image = get_one_image(transpose=(2, 0, 1))
    init.image = torch.from_numpy(image)


def run(config, Reporter):
    init(config)
    with Reporter() as reporter:
        reporter.run(config['exp_count'], init.model, init.image)
