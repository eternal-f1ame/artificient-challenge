"""
develop a model based on the onnx file in model/model.onnx 

Note:
    - initialize the convolutions layer with uniform xavier
    - initialize the linear layer with a normal distribution (mean=0.0, std=1.0)
    - initialize all biases with zeros
    - use batch norm wherever is relevant
    - use random seed 8
    - use default values for anything unspecified
"""

import numpy as np
import torch
import torch.nn as nn



torch.manual_seed(8)    # DO NOT MODIFY!
np.random.seed(8)   # DO NOT MODIFY!


# write your code here ...
import onnx
import onnxruntime as ort

onnx_model = onnx.load("model/model.onnx")
onnx.checker.check_model(onnx_model)
input_shape = [dim.dim_value for dim in onnx_model.graph.input[0].type.tensor_type.shape.dim]
output_shape = [dim.dim_value for dim in onnx_model.graph.output[0].type.tensor_type.shape.dim]

class CustomModel(nn.Module):
    def __init__(self, backbone = 'model/model.onnx'):
        super(CustomModel, self).__init__()
        self.sess = ort.InferenceSession(backbone)
        self.conv1 = nn.Conv2d(output_shape[1], 256, 7)
        self.conv2 = nn.Conv2d(256, 512, 7)
        self.pool = nn.MaxPool2d(4, 4)
        self.fc = nn.Linear(7168, 100)
        self.bn1 = nn.BatchNorm2d(256)
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.normal_(self.fc.weight, mean=0.0, std=1.0)
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)
        nn.init.zeros_(self.fc.bias)


    def forward(self, x):
        x = self.sess.run(None, {'input': x.cpu().numpy()
                                 })[0]
        x = torch.tensor(x).to('cuda')
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
model = CustomModel()

inputs = torch.randn(input_shape)
model = model.to('cuda')
inputs = inputs.to('cuda')
output = model(inputs)

# Show the output shape
print(output.shape)

# EOF
