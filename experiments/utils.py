from skimage import io
import numpy as np
from core.config import config
image_path = config['img_path']
image_class = config['img_class']


def get_one_image(transpose=None, img_type='numpy'):
    if img_type == 'numpy':
        numpy_img = io.imread(image_path).astype(dtype=np.float32) / 255
        if transpose:
            numpy_img = np.transpose(numpy_img, transpose)
        return np.expand_dims(numpy_img, axis=0), image_class
    else:
        return open(image_path, 'rb').read(), image_class
