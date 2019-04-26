import tensorflow as tf

from experiments.utils import get_one_image


def init(config):
    with tf.gfile.GFile(config['modelpath'], "rb") as f:
        restored_graph_def = tf.GraphDef()
        restored_graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            restored_graph_def,
            input_map=None,
            return_elements=None,
            name="")
    init.graph = graph
    init.image = get_one_image()
    init.images_tensor = graph.get_tensor_by_name('images:0')
    init.logits_tensor = graph.get_tensor_by_name('output:0')


def run(config, reporter):
    init(config)
    with tf.Session(graph=init.graph) as sess:
        sess.run([tf.global_variables_initializer()])
        with reporter:
            generator = reporter.run(
                config['exp_count'],
                sess.run,
                init.logits_tensor,
                feed_dict={init.images_tensor: init.image})
            for output in generator:
                assert output.shape == (1, 1001)
