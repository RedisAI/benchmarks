import requests

from experiments.utils import get_one_image


def init():
    init.img_list = get_one_image(transpose=(2, 0, 1)).tolist()


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
            # TODO: Perhaps raise the error later without crashing python
            assert len(output.json()['prediction'][0]) == 1000
