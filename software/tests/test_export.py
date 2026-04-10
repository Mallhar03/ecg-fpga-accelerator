import pytest
import torch
import os
import re
import json
import numpy as np
from src.models.quantized_cnn import QuantizedMultiScale1DCNN
from src.quantization.export import export_weights_to_mem

@pytest.fixture
def mock_config():
    """
    Minimal config for testing export logic.
    """
    return {
        'data': {
            'in_channels': 1, 
            'class_names': ['ST_segment', 'QT_interval', 'P_wave', 'Bundle_Branch_Block', 'Normal']
        },
        'model': {
            'num_classes': 5,
            'branch_out_channels': 8,
            'kernel_sizes': [3, 5],
            'pool_output_size': 4,
            'fc_hidden_size': 16,
            'dropout_rate': 0.1
        },
        'qat': {
            'weight_bits': 8,
            'activation_bits': 8
        },
        'export': {
            'output_dir': 'tmp_mem_test'
        }
    }

def test_export_creates_mem_files(mock_config, tmp_path):
    """
    Verify that .mem files and manifest are created in the output directory.
    """
    model = QuantizedMultiScale1DCNN(mock_config)
    output_dir = os.path.join(str(tmp_path), "mem_files")
    
    export_weights_to_mem(model, output_dir)
    
    assert os.path.exists(output_dir)
    files = os.listdir(output_dir)
    # Checks for at least one .mem file (conv or fc)
    assert any(f.endswith(".mem") for f in files)
    assert "weights_manifest.json" in files

def test_export_manifest_structure(mock_config, tmp_path):
    """
    Verify the fields in weights_manifest.json.
    """
    model = QuantizedMultiScale1DCNN(mock_config)
    output_dir = os.path.join(str(tmp_path), "mem_files")
    
    export_weights_to_mem(model, output_dir)
    
    manifest_path = os.path.join(output_dir, "weights_manifest.json")
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
        
    assert len(manifest) > 0
    for layer_name, info in manifest.items():
        assert "shape" in info
        assert "num_values" in info
        assert "scale_factor" in info
        assert "mem_file" in info
        assert isinstance(info['shape'], list)
        assert isinstance(info['num_values'], int)
        assert isinstance(info['scale_factor'], float)

def test_export_hex_format(mock_config, tmp_path):
    """
    Verify that each line in .mem files is exactly 2 uppercase hex characters.
    """
    model = QuantizedMultiScale1DCNN(mock_config)
    output_dir = os.path.join(str(tmp_path), "mem_files")
    
    export_weights_to_mem(model, output_dir)
    
    hex_regex = re.compile(r"^[0-9A-F]{2}$")
    
    mem_files = [f for f in os.listdir(output_dir) if f.endswith(".mem")]
    assert len(mem_files) > 0
    
    for f_name in mem_files:
        with open(os.path.join(output_dir, f_name), 'r') as mf:
            lines = mf.readlines()
            assert len(lines) > 0
            for line in lines:
                clean_line = line.strip()
                if clean_line:
                    assert hex_regex.match(clean_line), f"Invalid hex in {f_name}: {clean_line}"

def test_export_line_count_matches_manifest(mock_config, tmp_path):
    """
    Verify that the number of lines in each .mem file matches the 'num_values' in manifest.
    """
    model = QuantizedMultiScale1DCNN(mock_config)
    output_dir = os.path.join(str(tmp_path), "mem_files")
    
    export_weights_to_mem(model, output_dir)
    
    manifest_path = os.path.join(output_dir, "weights_manifest.json")
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
        
    for layer_name, info in manifest.items():
        mem_path = os.path.join(output_dir, info['mem_file'])
        with open(mem_path, 'r') as mf:
            # Count non-empty lines
            lines = [l for l in mf if l.strip()]
        assert len(lines) == info['num_values'], f"Line count mismatch for {layer_name}"
