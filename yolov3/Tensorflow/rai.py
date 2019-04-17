import os

import numpy as np
from PIL import Image
import redisai as rai
from redisai import model as raimodel

from core import reporter
from . import utils

def yolov3_tensorflow_rai(new_shape=416, count=10):
    basepath = os.path.dirname(os.path.dirname(__file__))
    img_path = os.path.join(basepath, 'data/img.jpg')
    graph_path = os.path.join(basepath, 'data/yolo.pb')
    pil_image = Image.open(img_path)
    numpy_img = np.array(pil_image)
    image_data = utils.letter_box(numpy_img, new_shape)

    graph = raimodel.Model.load(graph_path)

    inputs = ['input_1', 'input_image_shape']
    outputs = ['concat_11', 'concat_12', 'concat_13']

    con = rai.Client()
    con.modelset(
        'graph', rai.Backend.tf, rai.Device.cpu, graph,
        input=inputs, output=outputs)
    image = rai.BlobTensor.from_numpy(image_data)
    con.tensorset('image', image)
    input_shape = rai.Tensor(rai.DType.float, shape=[2], value=[new_shape, new_shape])
    con.tensorset('input_shape', input_shape)
    con.modelrun(
            'graph',
            input=['image', 'input_shape'],
            output=['boxes', 'scores', 'classes'])
    with reporter.Reporter() as r:
        r.runmore(count, con.modelrun, 'graph',
            input=['image', 'input_shape'],
            output=['boxes', 'scores', 'classes'])
    boxes = con.tensorget('boxes', as_type=rai.BlobTensor).to_numpy()
    classes = con.tensorget('classes', as_type=rai.BlobTensor).to_numpy()
    return r