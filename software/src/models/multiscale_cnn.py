import torch
import torch.nn as nn
import torch.nn.functional as F

import logging

logger = logging.getLogger("ecg_fpga.models.multiscale_cnn")

class MultiScale1DCNN(nn.Module):
    """
    Multi-Scale 1D-CNN for ECG Arrhythmia Detection.
    Optimized for FPGA acceleration.

    Architecture strictly matches hardware spec and Phase 2 QAT layout:
    - Three parallel branches, kernel sizes from config: [3, 5, 7]
    - Each branch: Conv1d → BatchNorm1d → ReLU → AdaptiveAvgPool1d(64)
    - Concat → Flatten → Linear(96*64, 128) → ReLU → Dropout → Linear(128, 5)
    - Output: raw logits shape (batch, 5)
    """
    def __init__(self, config):
        super(MultiScale1DCNN, self).__init__()
        
        in_channels = config['data'].get('in_channels', 1)
        num_classes = config['model']['num_classes']
        branch_out_channels = config['model']['branch_out_channels'] # 32
        kernel_sizes = config['model']['kernel_sizes'] # [3, 5, 7]
        pool_output_size = config['model']['pool_output_size'] # 64
        fc_hidden_size = config['model']['fc_hidden_size'] # 128
        dropout_rate = config['model']['dropout_rate'] # 0.3
        
        # Parallel Branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels, 
                    branch_out_channels, 
                    kernel_size=k, 
                    padding=k // 2
                ),
                nn.BatchNorm1d(branch_out_channels),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(pool_output_size)
            ) for k in kernel_sizes
        ])
        
        # Calculate flattened size
        flat_size = branch_out_channels * len(kernel_sizes) * pool_output_size
        
        # Concat -> Flatten -> FC1
        self.fc1 = nn.Linear(flat_size, fc_hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        
        # Output layer
        self.fc2 = nn.Linear(fc_hidden_size, num_classes)

        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"MultiScale1DCNN initialized with {total_params} parameters.")

    def forward(self, x):
        # x shape: [batch, 1, seq_len]
        branch_outs = [branch(x) for branch in self.branches]
        
        # Concatenate along channel dimension
        x = torch.cat(branch_outs, dim=1) 
        
        # Flatten
        x = torch.flatten(x, 1)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
