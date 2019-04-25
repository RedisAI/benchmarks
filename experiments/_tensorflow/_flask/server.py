from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf

# TODO: make it configurable
tf_model_path = '/root/data/resnet50.pb'

with tf.gfile.GFile(tf_model_path, "rb") as f:
    restored_graph_def = tf.GraphDef()
    restored_graph_def.ParseFromString(f.read())
with tf.Graph().as_default() as graph:
    tf.import_graph_def(
        restored_graph_def,
        input_map=None,
        return_elements=None,
        name="")


app = Flask(__name__)

sess = tf.Session(graph=graph)
images_tensor = graph.get_tensor_by_name('images:0')
logits_tensor = graph.get_tensor_by_name('output:0')


@app.route('/predict', methods=['POST'])
def predict():
    data = np.array(request.json['data'], dtype=np.float32)
    out = sess.run(logits_tensor, feed_dict={images_tensor: data})
    response = {'prediction': out.tolist()}
    return jsonify(response)
