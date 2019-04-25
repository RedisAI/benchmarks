from flask import Flask, request, jsonify
import numpy as np
import torch

# TODO: make it configurable
pt_model_path = '/root/data/resnet50.pt'


app = Flask(__name__)
model = torch.jit.load(pt_model_path)


@app.route('/predict', methods=['POST'])
def predict():
    data = np.array(request.json['data'], dtype=np.float32)
    torch_image = torch.from_numpy(data)
    with torch.no_grad():
        out = model(torch_image).numpy().tolist()
    response = {'prediction': out}
    return jsonify(response)
