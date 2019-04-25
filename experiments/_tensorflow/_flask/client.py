import requests

from experiments.utils import get_one_image


def init():
    init.img_list = get_one_image().tolist()


def run(config, reporter):
    # TODO: sending as list is not the best way
    init()
    with reporter:
        reporter.run(
            config['exp_count'],
            requests.post,
            config['server'],
            json={'data': init.img_list})
