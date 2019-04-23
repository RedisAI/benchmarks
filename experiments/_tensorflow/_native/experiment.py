import tensorflow as tf

from core.reporter import Reporter
from experiments.utils import get_one_image
from core.config import ConfigManager


with tf.gfile.GFile(ConfigManager.tf_model_path, "rb") as f:
    restored_graph_def = tf.GraphDef()
    restored_graph_def.ParseFromString(f.read())
with tf.Graph().as_default() as graph:
    tf.import_graph_def(
        restored_graph_def,
        input_map=None,
        return_elements=None,
        name="")

image = get_one_image()

images_tensor = graph.get_tensor_by_name('images:0')
logits_tensor = graph.get_tensor_by_name('output:0')


def run():
    with tf.Session(graph=graph) as sess:
        sess.run([tf.global_variables_initializer()])
        with Reporter() as reporter:
            reporter.run(
                ConfigManager.exp_count,
                sess.run,
                logits_tensor,
                feed_dict={images_tensor: image})
