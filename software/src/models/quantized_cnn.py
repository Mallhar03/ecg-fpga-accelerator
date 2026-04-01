import torch
import torch.nn as nn
import torch.nn.functional as F
from brevitas.nn import QuantConv1d, QuantLinear, QuantReLU, QuantIdentity
from brevitas.quant import Int8WeightPerTensorFloat, Int8ActPerTensorFloat

class QuantMultiScaleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, bit_width=8, kernel_sizes=[3, 5, 7, 9]):
        super(QuantMultiScaleLayer, self).__init__()
        self.convs = nn.ModuleList([
            QuantConv1d(
                in_channels, 
                out_channels // len(kernel_sizes), 
                kernel_size=k, 
                padding=k // 2,
                weight_quant=Int8WeightPerTensorFloat,
                weight_bit_width=bit_width,
                return_quant_tensor=True
            )
            for k in kernel_sizes
        ])
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = QuantReLU(act_quant=Int8ActPerTensorFloat, bit_width=bit_width, return_quant_tensor=True)

    def forward(self, x):
        features = [conv(x) for conv in self.convs]
        x = torch.cat([f.tensor if hasattr(f, 'tensor') else f for f in features], dim=1)
        x = self.bn(x)
        x = self.relu(x)
        return x

class QuantMultiScale1DCNN(nn.Module):
    """
    Quantized Multi-Scale 1D-CNN using Brevitas.
    Designed for deployment on PYNQ-Z1/Z2 via FINN or custom RTL.
    """
    def __init__(self, in_channels=1, num_classes=5, base_filters=16, bit_width=8):
        super(QuantMultiScale1DCNN, self).__init__()
        
        self.layer1 = QuantMultiScaleLayer(in_channels, base_filters, bit_width=bit_width)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.layer2 = QuantMultiScaleLayer(base_filters, base_filters * 2, bit_width=bit_width)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        self.layer3 = QuantMultiScaleLayer(base_filters * 2, base_filters * 4, bit_width=bit_width)
        self.pool3 = nn.AdaptiveAvgPool1d(1)
        
        self.fc = QuantLinear(
            base_filters * 4, 
            num_classes,
            weight_quant=Int8WeightPerTensorFloat,
            weight_bit_width=bit_width,
            bias=True
        )

    def forward(self, x):
        x = self.pool1(self.layer1(x))
        x = self.pool2(self.layer2(x))
        x = self.pool3(self.layer3(x))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
