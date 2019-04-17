# benchmarks

This repo aims to benchmark RedisAI against everything else which includes

- RedisAI with Tensorflow as backend vs Tensorflow python runtime
- RedisAI with PyTorch as backend vs PyTorch python runtime
- RedisAI with ONNX as backend vs ONNXRuntime (upcoming)
- RedisAI vs Tensorflow Serving (upcoming)
- RedisAI vs MxNet Model Server (upcoming)
- RedisAI vs TensorRT (upcoming)

Models used in for the benchmark will be autodownloaded if it doesn't exist in the `data` directory. However, if requires, it can be manually [downloaded](https://app.box.com/s/r4xzm4xtzdqhmg4rbwfcahj9tee3ojbl)


## Current Results

- OS Platform and Distribution: Linux Ubuntu 18.04
- Device: CPU
- Python version: 3.6
- Tensorflow version: 1.12.0
- TensorFlow optimizations: ON

|   Models  | TF Python  | TF RedisAI |
| --------- | ---------- | ---------- |
|   YoloV3  |   0.280    |   0.295    |
| ResNet50  |   0.038    |   0.041    |


- OS Platform and Distribution: Linux Ubuntu 18.04
- Device: CPU
- Python version: 3.6
- PyTorch: 1.0.1

|   Models  | PT Python  | PT RedisAI |
| --------- | ---------- | ---------- |
|   YoloV3  |   0.321    |   0.333    |
| ResNet50  |   0.062    |   0.082    |
