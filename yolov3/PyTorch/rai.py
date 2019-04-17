import os

from PIL import Image
import numpy as np
import redisai as rai
from redisai import model as raimodel

from core import reporter
from . import utils

def yolov3_pytorch_rai(new_shape=416, count=10):
	basepath = os.path.dirname(os.path.dirname(__file__))
	model_path = os.path.join(basepath, 'data/yolo.pt')
	image_path = os.path.join(basepath, 'data/img.jpg')
	pil_image = Image.open(image_path)
	numpy_image = np.array(pil_image)
	conf_thresh = 0.2
	nms_thresh = 0.5

	con = rai.Client()

	model = raimodel.Model.load(model_path)
	processed_image = utils.process_image(numpy_image, new_shape)
	image = rai.BlobTensor.from_numpy(processed_image)
	con.tensorset('image', image)
	con.modelset('model', rai.Backend.torch, rai.Device.cpu, model)
	con.modelrun('model', input=['image'], output=['out'])
	with reporter.Reporter() as r:
		r.runmore(
			count, con.modelrun, 'model', input=['image'], output=['out'])
	out = con.tensorget('out', as_type=rai.BlobTensor)
	return r

