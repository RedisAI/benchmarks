import numpy as np
from skimage import io
import json
import torch
import time

def resnet_pytorch_native(reporter):
	json_path = "../data/imagenet_classes.json"
	image_path = '../data/guitar.jpg'
	model_path = '../data/resnet50.pt'

	class_idx = json.load(open(json_path))

	model = torch.jit.load(model_path)
	# TODO: Fix it
	numpy_img = io.imread(image_path).astype(dtype=np.float32) / 255
	numpy_img = np.transpose(numpy_img, (2, 0, 1))
	numpy_img = np.expand_dims(numpy_img, axis=0)
	image = torch.from_numpy(numpy_img)
	out = model(image)
	a = time.time()
	out = model(image)
	print(time.time() - a)
	# print(out)

