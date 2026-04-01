import argparse
import sys
import os
import yaml
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from data.loader import ECGDataLoader
from models.multiscale_cnn import MultiScale1DCNN
from models.quantized_cnn import QuantMultiScale1DCNN
from training.trainer import ECGTrainer

def main():
    parser = argparse.ArgumentParser(description="Train ECG-FPGA Multi-Scale CNN")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--quantize", action="store_true", help="Train quantized model")
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data Loaders
    loader = ECGDataLoader(raw_dir=config['data']['raw_dir'], processed_dir=config['data']['processed_dir'])
    train_loader, val_loader = loader.get_loaders(
        batch_size=config['training']['batch_size'],
        test_split=config['data']['test_split']
    )

    # Initialize Model
    if args.quantize or config['quantization']['enabled']:
        print("Initializing Quantized Model (Brevitas)...")
        model = QuantMultiScale1DCNN(
            num_classes=config['model']['num_classes'],
            bit_width=config['quantization']['weight_bit_width']
        )
    else:
        print("Initializing FP32 Multi-Scale CNN Model...")
        model = MultiScale1DCNN(num_classes=config['model']['num_classes'])

    # Trainer
    trainer = ECGTrainer(model, config, train_loader, val_loader, device=device)
    trainer.train()

if __name__ == "__main__":
    main()
