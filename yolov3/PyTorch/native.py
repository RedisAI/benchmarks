import os

import torch
from PIL import Image
import numpy as np

from core import reporter
from . import utils

def yolov3_pytorch_native(new_shape=416, count=10):
	basepath = os.path.dirname(os.path.dirname(__file__))
	model_path = os.path.join(basepath, 'data/yolo.pt')
	image_path = os.path.join(basepath, 'data/img.jpg')
	pil_image = Image.open(image_path)
	numpy_image = np.array(pil_image)
	conf_thresh = 0.2
	nms_thresh = 0.5
	model = torch.jit.load(model_path)

	processed_image = torch.from_numpy(utils.process_image(numpy_image, new_shape))
	out = model(processed_image)
	with reporter.Reporter() as r:
		r.runmore(count, model, processed_image)
	out = utils.non_max_suppression(out, 80, conf_thresh, nms_thresh)
	return r
