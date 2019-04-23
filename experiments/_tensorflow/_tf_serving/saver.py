import tensorflow as tf
import tensorflow_hub as hub

url = 'https://tfhub.dev/google/imagenet/resnet_v2_50/classification/1'
images = tf.placeholder(tf.float32, shape=(1, 224, 224, 3), name='images')
module = hub.Module(url)
print(module.get_signature_names())
print(module.get_output_info_dict())
logits = module(images)
logits = tf.identity(logits, 'output')
export_path = 'build/1'
with tf.Session() as sess:
    sess.run([tf.global_variables_initializer()])
    images = sess.graph.get_tensor_by_name('images:0')
    tf.saved_model.simple_save(
        sess,
        export_path,
        inputs={'images': images},
        outputs={'output': logits})


# docker run -p 8500:8500 -p 8501:8501 --name tfserving \
#	--mount type=bind,source=/home/hhsecond/mypro/benchmarks/experiments/ServingTF/build,\
#	target=/models/resnet -e MODEL_NAME=resnet -t tensorflow/serving