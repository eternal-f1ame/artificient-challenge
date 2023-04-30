"""
Write a code using pytorch to replicate a grouped 2D convolution layer based on the original 2D convolution. 

The common way of using grouped 2D convolution layer in Pytorch is to use 
torch.nn.Conv2d(groups=n), where n is the number of groups.

However, it is possible to use a stack of n torch.nn.Conv2d(groups=1) to replicate the same
result. The wights must be copied and be split between the convs in the stack.

You can use:
    - use default values for anything unspecified  
    - all available functions in NumPy and Pytorch
    - the custom layer must be able to take all parameters of the original nn.Conv2d 
"""

import numpy as np
import torch
import torch.nn as nn


torch.manual_seed(8)    # DO NOT MODIFY!
np.random.seed(8)   # DO NOT MODIFY!

# random input (batch, channels, height, width)
x = torch.randn(2, 64, 100, 100)

# original 2d convolution
grouped_layer = nn.Conv2d(64, 128, 3, stride=1, padding=1, groups=16, bias=True)

# weights and bias
w_torch = grouped_layer.weight
b_torch = grouped_layer.bias

y = grouped_layer(x)

# now write your custom layer
class Conv2dGrouped(nn.Module):
    """Grouped 2D convolution layer based on the original 2D convolution
    # Arguments
        `in_channels`: Integer, the channels of the input tensor\n
        `out_channels`: Integer, the channels of the output tensor\n
        `kernel_size`: Integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window\n
        `stride`: Integer or tuple/list of 2 integers, specifying the strides of the convolution along the height and width\n
        `padding`: Integer or tuple/list of 2 integers, specifying the padding of the input\n
        `dilation`: Integer or tuple/list of 2 integers, specifying the dilation rate to use for dilated convolution\n
        `groups`: Integer, specifying the number of groups in which the input is split along the channel axis\n
        `bias`: Boolean, whether the layer uses a bias vector
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dGrouped, self).__init__()

        assert out_channels % groups == 0, f"Number of output channels {out_channels} must be divisible by number of groups {groups}"
        assert in_channels % groups == 0, f"Number of input channels {in_channels} must be divisible by number of groups {groups}"

        self.groups = groups
        self.convs = nn.ModuleList()
        self.split = lambda x: torch.split(x, x.size(1) // self.groups, dim=1)

        for _ in range(groups):
            self.convs.append(
                nn.Conv2d(
                in_channels//groups,
                out_channels//groups,
                kernel_size,
                stride,
                padding,
                dilation,
                groups=1,
                bias=bias)
                )

    def forward(self, x):
        x_split = self.split(x)
        out = torch.cat([conv(x_split[i]) for i, conv in enumerate(self.convs)], dim=1)
        return out

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

# the output of CustomGroupedConv2D(x) must be equal to grouped_layer(x)
grouped_layer_custom = Conv2dGrouped(64, 128, 3, stride=1, padding=1, groups=16, bias=True)

# copy weights and bias
for i in range(grouped_layer_custom.groups):
    grouped_layer_custom.convs[i].weight = nn.Parameter(w_torch[i*(128//16):(i+1)*(128//16)])
    grouped_layer_custom.convs[i].bias = nn.Parameter(b_torch[i*(128//16):(i+1)*(128//16)])

y_custom = grouped_layer_custom(x)

# Show that the outputs shapes are equal
print(y.shape)
print(y_custom.shape)

# Print the difference between the outputs
print(f"Difference is : {torch.sum(y - y_custom)}")

# EOF
