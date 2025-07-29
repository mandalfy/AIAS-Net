import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import numpy as np
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt

# Import AIAS-Net components
from aias_net import AIASNet
from config import Config, get_config
from data_loader import get_data_loaders, analyze_class_distribution

class AIASNetTrainer:
    def __init__(self, config, save_dir="./checkpoints"):
        self.config = config
        self.save_dir = save_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize model
        self.model = AIASNet(
            num_classes=config.data_config["num_classes"], 
            config=config
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = self._get_optimizer()
        
        # Initialize scheduler
        self.scheduler = self._get_scheduler()
        
        # Training history
        self.train_history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'loss_components': []
        }
        
        print(f"Initialized AIAS-Net Trainer on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _get_optimizer(self):
        """Initialize optimizer based on configuration."""
        optimizer_name = self.config.training_config["optimizer"]
        lr = self.config.training_config["learning_rate"]
        weight_decay = self.config.training_config["weight_decay"]
        
        if optimizer_name.lower() == "adam":
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name.lower() == "sgd":
            return optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    def _get_scheduler(self):
        """Initialize learning rate scheduler."""
        scheduler_name = self.config.training_config.get("scheduler", "StepLR")
        
        if scheduler_name == "StepLR":
            step_size = self.config.training_config.get("step_size", 30)
            gamma = self.config.training_config.get("gamma", 0.1)
            return StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        else:
            return None
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        loss_components_epoch = []
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs, loss = self.model(images, masks)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Get individual loss components for meta-learning
            if hasattr(self.model.hialf, 'get_individual_losses'):
                loss_components = self.model.hialf.get_individual_losses(outputs, masks)
                loss_components_epoch.append(loss_components)
                
                # Meta-learning update (simplified)
                if len(loss_components_epoch) > 1:
                    support_losses = loss_components_epoch[-2]
                    query_losses = loss_components_epoch[-1]
                    
                    try:
                        meta_loss = self.model.meta_learner.meta_update(
                            support_losses, query_losses, 
                            self.model.das, self.model.hialf
                        )
                    except Exception as e:
                        print(f"Meta-learning update failed: {e}")
            
            # Print progress
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        return avg_loss, loss_components_epoch
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(val_loader)
        
        # Metrics
        total_correct = 0
        total_pixels = 0
        class_correct = torch.zeros(self.config.data_config["num_classes"])
        class_total = torch.zeros(self.config.data_config["num_classes"])
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs, loss = self.model(images, masks)
                total_loss += loss.item()
                
                # Calculate metrics
                predictions = torch.argmax(outputs, dim=1) if outputs.shape[1] > 1 else (torch.sigmoid(outputs) > 0.5).long().squeeze(1)
                
                # Overall accuracy
                correct = (predictions == masks).sum().item()
                total_correct += correct
                total_pixels += masks.numel()
                
                # Class-specific accuracy
                for c in range(self.config.data_config["num_classes"]):
                    class_mask = (masks == c)
                    class_correct[c] += (predictions[class_mask] == c).sum().item()
                    class_total[c] += class_mask.sum().item()
        
        avg_loss = total_loss / num_batches
        overall_accuracy = total_correct / total_pixels
        
        # Calculate class-specific metrics
        class_accuracies = []
        for c in range(self.config.data_config["num_classes"]):
            if class_total[c] > 0:
                class_acc = class_correct[c] / class_total[c]
                class_accuracies.append(class_acc)
            else:
                class_accuracies.append(0.0)
        
        metrics = {
            'overall_accuracy': overall_accuracy,
            'class_accuracies': class_accuracies,
            'mean_class_accuracy': np.mean(class_accuracies)
        }
        
        return avg_loss, metrics
    
    def train(self, train_loader, val_loader):
        """Main training loop."""
        num_epochs = self.config.training_config["num_epochs"]
        best_val_loss = float('inf')
        
        print(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)
            
            # Training
            train_loss, loss_components = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_metrics = self.validate_epoch(val_loader)
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            
            # Save history
            self.train_history['epoch'].append(epoch + 1)
            self.train_history['train_loss'].append(train_loss)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_metrics'].append(val_metrics)
            self.train_history['loss_components'].append(loss_components)
            
            # Print epoch results
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Accuracy: {val_metrics['overall_accuracy']:.4f}")
            print(f"Mean Class Accuracy: {val_metrics['mean_class_accuracy']:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, is_best=True)
                print("Saved best model!")
            
            # Save regular checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch)
        
        print("\nTraining completed!")
        self.plot_training_history()
        return self.train_history
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_history': self.train_history,
            'config': self.config.__dict__
        }
        
        if is_best:
            checkpoint_path = os.path.join(self.save_dir, 'best_model.pth')
        else:
            checkpoint_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch + 1}.pth')
        
        torch.save(checkpoint, checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_history = checkpoint['train_history']
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint['epoch']
    
    def plot_training_history(self):
        """Plot training history."""
        epochs = self.train_history['epoch']
        train_losses = self.train_history['train_loss']
        val_losses = self.train_history['val_loss']
        
        plt.figure(figsize=(12, 4))
        
        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, label='Train Loss')
        plt.plot(epochs, val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        # Accuracy plot
        plt.subplot(1, 2, 2)
        val_accuracies = [m['overall_accuracy'] for m in self.train_history['val_metrics']]
        mean_class_accuracies = [m['mean_class_accuracy'] for m in self.train_history['val_metrics']]
        
        plt.plot(epochs, val_accuracies, label='Overall Accuracy')
        plt.plot(epochs, mean_class_accuracies, label='Mean Class Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Validation Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_history.png'))
        plt.show()

def main():
    """Main training function."""
    # Get configuration
    config = get_config("experiment", "high_imbalance")  # Use high imbalance configuration
    
    # Create data loaders
    data_loaders = get_data_loaders(config)
    
    # Analyze dataset
    print("Analyzing dataset...")
    train_stats = analyze_class_distribution(data_loaders['train'], config.data_config["num_classes"])
    print(f"Class distribution: {train_stats}")
    
    # Initialize trainer
    trainer = AIASNetTrainer(config, save_dir="./aias_net_checkpoints")
    
    # Start training
    history = trainer.train(data_loaders['train'], data_loaders['val'])
    
    # Save final results
    with open(os.path.join(trainer.save_dir, 'training_results.json'), 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_history = {}
        for key, value in history.items():
            if isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], np.ndarray):
                    serializable_history[key] = [v.tolist() for v in value]
                else:
                    serializable_history[key] = value
            else:
                serializable_history[key] = value
        
        json.dump(serializable_history, f, indent=2)
    
    print(f"Training results saved to {trainer.save_dir}")

if __name__ == "__main__":
    main()

