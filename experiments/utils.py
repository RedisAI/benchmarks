from skimage import io
import numpy as np
from core.config import ConfigManager

image_path = ConfigManager.image_path


def get_one_image(transpose=None):
    numpy_img = io.imread(image_path).astype(dtype=np.float32) / 255
    if transpose:
        numpy_img = np.transpose(numpy_img, transpose)
    return np.expand_dims(numpy_img, axis=0)
