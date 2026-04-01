#!/usr/bin/env python3
"""
Download MIT-BIH Arrhythmia Database records from PhysioNet.

Usage:
    # Download specific records:
    python software/scripts/download_dataset.py --output software/data/raw/ --records 100 101 102

    # Download all records defined in config:
    python software/scripts/download_dataset.py --output software/data/raw/ --all

Requires: free PhysioNet account registration at https://physionet.org
"""

import argparse
import logging
import os
import sys
import yaml
from tqdm import tqdm
import wfdb

# CLI scripts configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Download MIT-BIH records from PhysioNet")
    parser.add_argument(
        "--output",
        required=True,
        help="Directory to save downloaded records"
    )
    parser.add_argument(
        "--records",
        nargs="+",
        help="Specific record IDs to download"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all records defined in config (train + test)"
    )
    parser.add_argument(
        "--config",
        default="software/config/config.yaml",
        help="Path to config.yaml (default: software/config/config.yaml)"
    )

    args = parser.parse_args()

    # Determine record IDs to download
    records_to_download = []
    
    if args.all:
        if not os.path.exists(args.config):
            logger.error(f"Config file not found: {args.config}")
            sys.exit(1)
        
        with open(args.config, 'r') as f:
            try:
                config = yaml.safe_load(f)
                train = config.get('data', {}).get('train_records', [])
                test = config.get('data', {}).get('test_records', [])
                records_to_download = train + test
            except yaml.YAMLError as exc:
                logger.error(f"Error parsing config.yaml: {exc}")
                sys.exit(1)
    elif args.records:
        records_to_download = args.records
    else:
        parser.print_usage()
        print("Error: either --all or --records ID1 ID2 ... must be specified.")
        sys.exit(1)

    # Filter out empty strings if any
    records_to_download = [r for r in records_to_download if r]

    if not records_to_download:
        logger.error("No records specified for download.")
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Download records
    success_count = 0
    for record_id in tqdm(records_to_download, desc="Downloading records"):
        try:
            wfdb.dl_files(
                'mitdb', 
                args.output, 
                [f'{record_id}.hea', f'{record_id}.dat', f'{record_id}.atr']
            )
            success_count += 1
        except Exception as e:
            logger.error(f"Failed to download record {record_id}: {str(e)}")

    # Final summary line
    print(f"Downloaded {success_count}/{len(records_to_download)} records to {args.output}.")

if __name__ == "__main__":
    main()
