import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

class MorphologyWeightedBCELoss(nn.Module):
    """Custom loss for weighting ECG morphological features."""
    def __init__(self, pos_weight=None):
        super(MorphologyWeightedBCELoss, self).__init__()
        self.bce = nn.CrossEntropyLoss(weight=pos_weight)

    def forward(self, outputs, targets):
        return self.bce(outputs, targets)

class ECGTrainer:
    def __init__(self, model, config, train_loader, val_loader, device='cpu'):
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        self.criterion = MorphologyWeightedBCELoss()
        
        os.makedirs(config['training']['checkpoint_dir'], exist_ok=True)
        os.makedirs(config['training']['log_dir'], exist_ok=True)
        os.makedirs(config['training']['plot_dir'], exist_ok=True)

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(tqdm(self.train_loader)):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
        return running_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                running_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        accuracy = 100. * correct / total
        return running_loss / len(self.val_loader), accuracy

    def train(self):
        epochs = self.config['training']['epochs']
        best_acc = 0.0
        
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch()
            val_loss, val_acc = self.validate()
            
            print(f"Epoch {epoch}/{epochs}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), os.path.join(self.config['training']['checkpoint_dir'], 'best_model.pth'))
        
        print(f"Training complete. Best Accuracy: {best_acc:.2f}%")
