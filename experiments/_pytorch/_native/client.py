import torch
from experiments.utils import get_one_image


def init(config):
    init.model = torch.jit.load(config['modelpath'])
    image, init.img_class = get_one_image(transpose=(2, 0, 1))
    init.image = torch.from_numpy(image)


def wrapper(init):
    with torch.no_grad():
        out = init.model(init.image)
    return out.numpy()


def run(config, reporter):
    init(config)
    with reporter:
        generator = reporter.run(config['exp_count'], wrapper, init)
        for output in generator:
            assert output.shape == (1, 1000)
            assert output.argmax() == init.img_class
