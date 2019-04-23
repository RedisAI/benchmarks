import torch

from core.reporter import Reporter
from experiments.utils import get_one_image
from core.config import ConfigManager

model = torch.jit.load(ConfigManager.pt_model_path)
transpose = (2, 0, 1)
image = get_one_image(transpose=transpose)
image = torch.from_numpy(image)


def run():
    with Reporter() as reporter:
        reporter.run(
            ConfigManager.exp_count,
            model, image)
