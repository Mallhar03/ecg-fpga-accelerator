import torch
import torch.nn as nn
from brevitas.nn import QuantConv1d, QuantLinear
from brevitas.quant import Int8WeightPerTensorFloat, Int8ActPerTensorFloat
import logging

logger = logging.getLogger("ecg_fpga.models.quantized_cnn")

class QuantizedMultiScale1DCNN(nn.Module):
    """
    INT8 quantized version of MultiScale1DCNN using Brevitas QAT.

    Architecture mirrors MultiScale1DCNN exactly:
    - Three parallel branches, kernel sizes from config: [3, 5, 7]
    - Each branch: QuantConv1d → BatchNorm1d → ReLU → AdaptiveAvgPool1d(64)
    - Concat → Flatten → QuantLinear(96*64, 128) → ReLU → Dropout → QuantLinear(128, 5)
    - Output: raw logits shape (batch, 5) — NO sigmoid

    All quantization parameters from config:
        config['qat']['weight_bits'] = 8
        config['qat']['activation_bits'] = 8

    Args:
        config (dict): Full parsed config.yaml dict.
    """
    def __init__(self, config):
        super(QuantizedMultiScale1DCNN, self).__init__()
        
        in_channels = config['data'].get('in_channels', 1)
        num_classes = config['model']['num_classes']
        branch_out_channels = config['model']['branch_out_channels'] # 32
        kernel_sizes = config['model']['kernel_sizes'] # [3, 5, 7]
        pool_output_size = config['model']['pool_output_size'] # 64
        fc_hidden_size = config['model']['fc_hidden_size'] # 128
        dropout_rate = config['model']['dropout_rate'] # 0.3
        
        weight_bits = config['qat']['weight_bits']
        act_bits = config['qat']['activation_bits']

        # Parallel Branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                QuantConv1d(
                    in_channels, 
                    branch_out_channels, 
                    kernel_size=k, 
                    padding=k // 2,
                    weight_bit_width=weight_bits,
                    weight_quant=Int8WeightPerTensorFloat,
                    input_bit_width=act_bits,
                    input_quant=Int8ActPerTensorFloat,
                    return_quant_tensor=True
                ),
                nn.BatchNorm1d(branch_out_channels),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(pool_output_size)
            ) for k in kernel_sizes
        ])
        
        # Calculate flattened size
        flat_size = branch_out_channels * len(kernel_sizes) * pool_output_size
        
        # Concat -> Flatten -> FC1
        self.fc1 = QuantLinear(
            flat_size, 
            fc_hidden_size,
            weight_bit_width=weight_bits,
            weight_quant=Int8WeightPerTensorFloat,
            input_bit_width=act_bits,
            input_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        
        # Last layer returns plain tensor (not QuantTensor) as requested
        self.fc2 = QuantLinear(
            fc_hidden_size, 
            num_classes,
            weight_bit_width=weight_bits,
            weight_quant=Int8WeightPerTensorFloat,
            input_bit_width=act_bits,
            input_quant=Int8ActPerTensorFloat,
            return_quant_tensor=False
        )

        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"QuantizedMultiScale1DCNN initialized with {total_params} parameters.")

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
