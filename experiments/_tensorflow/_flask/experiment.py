import requests

from core.reporter import Reporter
from experiments.utils import get_one_image
from core.config import ConfigManager

img_list = get_one_image().tolist()


def run():
    # TODO: sending as list is not the best way
    with Reporter() as reporter:
        reporter.run(
            ConfigManager.exp_count,
            requests.post,
            ConfigManager.flask_pytorch_url,
            json={'data': img_list})
