import requests

import numpy as np
from experiments.utils import get_one_image


def init():
    init.img_list, init.img_class = get_one_image()
    init.img_list = init.img_list.tolist()


def run(config, reporter):
    # TODO: sending as list is not the best way
    init()
    with reporter:
        generator = reporter.run(
            config['exp_count'],
            requests.post,
            config['server'],
            json={'data': init.img_list})
        for output in generator:
            output_list = output.json()['prediction'][0]
            assert len(output_list) == 1001
            assert np.argmax(output_list) - 1 == init.img_class
