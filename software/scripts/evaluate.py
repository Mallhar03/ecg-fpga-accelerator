import argparse
import sys
import os
import yaml
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from data.loader import ECGDataLoader
from models.multiscale_cnn import MultiScale1DCNN
from models.quantized_cnn import QuantMultiScale1DCNN

def plot_confusion_matrix(cm, classes, plot_dir):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    # Add counts to cells
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, 'confusion_matrix.png'))
    print(f"Confusion matrix saved to {plot_dir}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate ECG-FPGA Model")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pth")
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data Loaders
    loader = ECGDataLoader(raw_dir=config['data']['raw_dir'], processed_dir=config['data']['processed_dir'])
    # Process all records for evaluation to ensure all classes are present
    loader.process_data(force=False) 
    
    _, val_loader = loader.get_loaders(
        batch_size=config['training']['batch_size'],
        test_split=config['data']['test_split']
    )

    # Initialize and Load Model
    if config['quantization']['enabled']:
        model = QuantMultiScale1DCNN(
            num_classes=config['model']['num_classes'],
            bit_width=config['quantization']['weight_bit_width']
        )
    else:
        model = MultiScale1DCNN(num_classes=config['model']['num_classes'])
    
    checkpoint_path = args.checkpoint
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint {checkpoint_path} not found.")
        return

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []

    print("Starting evaluation...")
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    # Report
    classes = ['N', 'S', 'V', 'F', 'Q']
    present_labels = np.unique(all_targets)
    present_classes = [classes[i] for i in present_labels if i < len(classes)]
    
    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds, labels=present_labels, target_names=present_classes))

    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    plot_confusion_matrix(cm, present_classes, config['training']['plot_dir'])

if __name__ == "__main__":
    main()
