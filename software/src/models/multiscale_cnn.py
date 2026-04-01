import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7, 9]):
        super(MultiScaleLayer, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels // len(kernel_sizes), kernel_size=k, padding=k // 2)
            for k in kernel_sizes
        ])
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        features = [conv(x) for conv in self.convs]
        x = torch.cat(features, dim=1)
        x = F.relu(self.bn(x))
        return x

class MultiScale1DCNN(nn.Module):
    """
    Multi-Scale 1D-CNN for ECG Arrhythmia Detection.
    Optimized for FPGA acceleration.
    """
    def __init__(self, in_channels=1, num_classes=5, base_filters=16):
        super(MultiScale1DCNN, self).__init__()
        
        self.layer1 = MultiScaleLayer(in_channels, base_filters)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.layer2 = MultiScaleLayer(base_filters, base_filters * 2)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        self.layer3 = MultiScaleLayer(base_filters * 2, base_filters * 4)
        self.pool3 = nn.AdaptiveAvgPool1d(1)
        
        self.fc = nn.Linear(base_filters * 4, num_classes)

    def forward(self, x):
        # x shape: [batch, 1, seq_len]
        x = self.pool1(self.layer1(x))
        x = self.pool2(self.layer2(x))
        x = self.pool3(self.layer3(x))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
