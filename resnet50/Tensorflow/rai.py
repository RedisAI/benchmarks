from skimage import io
import numpy as np
import json
import time
import redisai as rai
from redisai import model as raimodel

json_path = "../data/imagenet_classes.json"
image_path = '../data/guitar.jpg'
class_idx = json.load(open(json_path))
numpy_img = io.imread(image_path).astype(dtype=np.float32)
numpy_img = np.expand_dims(numpy_img, axis=0) / 255
graph_path = "../data/resnet50.pb"

con = rai.Client()
graph = raimodel.Model.load(graph_path)

inputs = ['images']
outputs = ['output']

image = rai.BlobTensor.from_numpy(numpy_img)
con.tensorset('image', image)
con.modelset(
	'graph', rai.Backend.tf, rai.Device.cpu, graph,
	input=inputs, output=outputs)
con.modelrun('graph', input=['image'], output=['output'])
a = time.time()
con.modelrun('graph', input=['image'], output=['output'])
print(time.time() - a)
# print(con.tensorget('output').value)
