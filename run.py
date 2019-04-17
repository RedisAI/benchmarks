from yolov3.PyTorch.native import yolov3_pytorch_native
from yolov3.PyTorch.rai import yolov3_pytorch_rai
from yolov3.Tensorflow.native import yolov3_tensorflow_native
from yolov3.Tensorflow.rai import yolov3_tensorflow_rai

report = yolov3_pytorch_native()
report.summary()
report = yolov3_pytorch_rai()
report.summary()
report = yolov3_tensorflow_native()
report.summary()
report = yolov3_tensorflow_rai()
report.summary()
