import cv2
import numpy as np
import tensorflow as tf


def load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)
    return graph


def letter_box(numpy_image, height):
    shape = numpy_image.shape[:2]
    ratio = float(height) / max(shape)
    new_shape = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))
    dw = (height - new_shape[0]) / 2
    dh = (height - new_shape[1]) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    numpy_image = cv2.resize(numpy_image, new_shape, interpolation=cv2.INTER_AREA)
    numpy_image = cv2.copyMakeBorder(
        numpy_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(127.5, 127.5, 127.5))
    return np.expand_dims(numpy_image, axis=0).astype(np.float32) / 255
