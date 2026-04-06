import torch
import torch.nn as nn
import numpy as np
import os
import json
from brevitas.nn import QuantConv1d, QuantLinear

def export_weights_to_mem(quantized_model: torch.nn.Module,
                           output_dir: str) -> dict:
    """
    Extract INT8 weights from quantized model and write .mem files
    for hardware handoff to Sanjana.

    For each QuantConv1d and QuantLinear layer:
        1. Extract integer weights using layer.weight.int_repr()
        2. Convert to numpy int8 array
        3. Flatten in row-major (C) order
        4. Write one value per line as 2-digit uppercase hex (two's complement)
        5. Filename: layer_{layer_name}_weights.mem

    Also writes weights_manifest.json containing for each layer:
        - shape (original tensor shape as list)
        - num_values (total number of int8 values)
        - scale_factor (float, from layer.weight.quant_scale().item())
        - mem_file (filename string)

    Args:
        quantized_model: Trained QuantizedMultiScale1DCNN instance
        output_dir: Directory to write .mem files and manifest

    Returns:
        dict: The manifest dict (same as written to JSON)

    Raises:
        ValueError: If quantized_model has no quantized layers found
        OSError: If output_dir cannot be created
    """
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            raise OSError(f"Could not create output directory {output_dir}: {e}")

    manifest = {}
    quant_layers_found = 0
    
    # Ensure model is on CPU for weight extraction
    quantized_model.cpu()
    
    for name, module in quantized_model.named_modules():
        if isinstance(module, (QuantConv1d, QuantLinear)):
            quant_layers_found += 1
            
            # 1. Extract integer weights
            # quant_weight() returns a QuantTensor
            qw = module.quant_weight()
            # Calculate integer representation: round(value / scale)
            int_weights_tensor = torch.round(qw.value / qw.scale).detach()
            scale = qw.scale.item()
            
            # 2. Convert to numpy int8 array
            int_weights = int_weights_tensor.numpy().astype(np.int8)
            
            # 3. Flatten in row-major (C) order
            flat_weights = int_weights.flatten(order='C')
            
            # 4. Convert to two's complement hex
            # We use a uint8 view to get the unsigned byte value for hex printing
            uint8_weights = flat_weights.view(np.uint8)
            
            mem_filename = f"layer_{name.replace('.', '_')}_weights.mem"
            mem_path = os.path.join(output_dir, mem_filename)
            
            # 5. Write to .mem file (uppercase hex, zero-padded, no blank lines)
            with open(mem_path, 'w') as f:
                for val in uint8_weights:
                    f.write(f"{val:02X}\n")
            
            manifest[name] = {
                "shape": list(int_weights.shape),
                "num_values": int(flat_weights.size),
                "scale_factor": float(scale),
                "mem_file": mem_filename
            }

    if quant_layers_found == 0:
        raise ValueError("No quantized layers (QuantConv1d, QuantLinear) found in model.")
        
    manifest_path = os.path.join(output_dir, "weights_manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
        
    return manifest
