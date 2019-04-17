import numpy as np
from skimage import io
import json
import time
import redisai as rai
from redisai import model as raimodel

json_path = "../data/imagenet_classes.json"
image_path = '../data/guitar.jpg'
model_path = '../data/resnet50.pt'

class_idx = json.load(open(json_path))
model = raimodel.Model.load(model_path)
con = rai.Client()
numpy_img = io.imread(image_path).astype(dtype=np.float32) / 255
numpy_img = np.transpose(numpy_img, (2, 0, 1))
numpy_img = np.expand_dims(numpy_img, axis=0)
image = rai.BlobTensor.from_numpy(numpy_img)
con.tensorset('image', image)
con.modelset('model', rai.Backend.torch, rai.Device.cpu, model)
con.modelrun('model', input=['image'], output=['out'])
a = time.time()
con.modelrun('model', input=['image'], output=['out'])
print(time.time() - a)
# val = con.tensorget('out').value
# print(val)
