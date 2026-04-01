import argparse
import sys
import os
import yaml

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from data.loader import ECGDataLoader

def main():
    parser = argparse.ArgumentParser(description="Download MIT-BIH Arrhythmia Dataset")
    parser.add_argument("--output", type=str, default="ecg-fpga-accelerator/software/data/raw", help="Output directory")
    parser.add_argument("--all", action="store_true", help="Download all records")
    args = parser.parse_args()

    # Load config to get defaults if needed
    config_path = os.path.join(os.path.dirname(__file__), '../config/config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    loader = ECGDataLoader(raw_dir=args.output, processed_dir=config['data']['processed_dir'])
    
    if args.all:
        loader.download_mitdb()
    else:
        print("Please specific --all to download the dataset.")

if __name__ == "__main__":
    main()
