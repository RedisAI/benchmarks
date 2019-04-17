import os

import tensorflow as tf
import numpy as np
from PIL import Image

from core import reporter
from . import utils

def yolov3_tensorflow_native(new_shape=416, count=10):
    new_shape = 416
    basepath = os.path.dirname(os.path.dirname(__file__))
    img_path = os.path.join(basepath, 'data/img.jpg')
    graph_path = os.path.join(basepath, 'data/yolo.pb')
    pil_image = Image.open(img_path)
    numpy_img = np.array(pil_image)
    image_data = utils.letter_box(numpy_img, new_shape)
    image_shape_data = np.array([new_shape, new_shape])

    graph = utils.load_graph(graph_path)

    # output
    boxes = graph.get_tensor_by_name('import/concat_11:0')
    scores = graph.get_tensor_by_name('import/concat_12:0')
    classes = graph.get_tensor_by_name('import/concat_13:0')

    # input
    images = graph.get_tensor_by_name('import/input_1:0')
    image_shape = graph.get_tensor_by_name('import/input_image_shape:0')


    with tf.Session(graph=graph) as sess:
        # Dummy run to load TF
        out_boxes, out_scores, out_classes = sess.run(
            [boxes, scores, classes],
            feed_dict={images: image_data, image_shape: image_shape_data})
        r = reporter.Reporter()
        r.runmore(count, sess.run, [boxes, scores, classes],
            feed_dict={images: image_data, image_shape: image_shape_data})
        r.close()
    return r
