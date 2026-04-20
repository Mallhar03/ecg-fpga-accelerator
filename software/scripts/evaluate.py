import argparse
import yaml
import sys
import os
import logging
import torch
import numpy as np
import json
from tqdm import tqdm

# Ensure software directory is in path for imports
sys.path.append(os.path.join(os.getcwd(), 'software'))

from src.data.dataset import make_dataloaders
from src.models.multiscale_cnn import MultiScale1DCNN
from src.training.metrics import compute_clinical_metrics, CLASS_NAMES

def main():
    parser = argparse.ArgumentParser(description="Evaluate ECG-FPGA Accelerator Model")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pth")
    args = parser.parse_args()

    # Logging setup for CLI
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )
    logger = logging.getLogger("ecg_fpga.scripts.evaluate")

    try:
        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # Data - Using the same loading logic as train.py (conceptually)
        # Assuming make_dataloaders handles test data
        _, _, test_loader = make_dataloaders(config)

        # Model
        model = MultiScale1DCNN(config).to(device)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        all_logits = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Evaluating"):
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images)
                all_logits.append(logits.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        all_logits = np.concatenate(all_logits, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        # Apply sigmoid to logits for metrics
        probs = 1 / (1 + np.exp(-all_logits))

        metrics = compute_clinical_metrics(all_labels, probs, threshold=config['inference']['threshold'])

        # Print formatted table
        print(f"\n{'Class':<25} {'Sensitivity':<12} {'Specificity':<12} {'F1':<12}")
        print("-" * 65)
        for class_name in CLASS_NAMES:
            m = metrics[class_name]
            print(f"{class_name:<25} {m['sensitivity']:<12.4f} {m['specificity']:<12.4f} {m['f1']:<12.4f}")
        
        m_avg = metrics['macro_avg']
        print("-" * 65)
        print(f"{'Macro Average':<25} {m_avg['sensitivity']:<12.4f} {m_avg['specificity']:<12.4f} {m_avg['f1']:<12.4f}\n")

        # Save to JSON
        output_dir = "software/outputs/plots"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "test_metrics.json")
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Metrics saved to {output_path}")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
