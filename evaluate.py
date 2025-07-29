import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os
import json
from PIL import Image

# Import AIAS-Net components
from aias_net import AIASNet
from config import Config, get_config
from data_loader import get_data_loaders

class AIASNetEvaluator:
    def __init__(self, config, model_path, save_dir="./evaluation_results"):
        self.config = config
        self.save_dir = save_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize and load model
        self.model = AIASNet(
            num_classes=config.data_config["num_classes"], 
            config=config
        ).to(self.device)
        
        self.load_model(model_path)
        self.model.eval()
        
        print(f"Initialized AIAS-Net Evaluator on device: {self.device}")
    
    def load_model(self, model_path):
        """Load trained model."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            self.model.load_state_dict(checkpoint)
            print("Loaded model state dict")
    
    def calculate_metrics(self, predictions, targets, num_classes):
        """Calculate comprehensive evaluation metrics."""
        # Flatten predictions and targets
        pred_flat = predictions.flatten()
        target_flat = targets.flatten()
        
        # Overall metrics
        overall_accuracy = (pred_flat == target_flat).sum().item() / len(pred_flat)
        
        # Class-specific metrics
        class_metrics = {}
        for c in range(num_classes):
            class_mask = (target_flat == c)
            if class_mask.sum() > 0:
                class_pred = pred_flat[class_mask]
                class_accuracy = (class_pred == c).sum().item() / len(class_pred)
                
                # Precision, Recall, F1
                tp = ((pred_flat == c) & (target_flat == c)).sum().item()
                fp = ((pred_flat == c) & (target_flat != c)).sum().item()
                fn = ((pred_flat != c) & (target_flat == c)).sum().item()
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                
                # IoU (Intersection over Union)
                intersection = tp
                union = tp + fp + fn
                iou = intersection / union if union > 0 else 0.0
                
                class_metrics[f'class_{c}'] = {
                    'accuracy': class_accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'iou': iou,
                    'support': class_mask.sum().item()
                }
        
        # Mean metrics across classes
        mean_accuracy = np.mean([m['accuracy'] for m in class_metrics.values()])
        mean_precision = np.mean([m['precision'] for m in class_metrics.values()])
        mean_recall = np.mean([m['recall'] for m in class_metrics.values()])
        mean_f1 = np.mean([m['f1'] for m in class_metrics.values()])
        mean_iou = np.mean([m['iou'] for m in class_metrics.values()])
        
        return {
            'overall_accuracy': overall_accuracy,
            'mean_accuracy': mean_accuracy,
            'mean_precision': mean_precision,
            'mean_recall': mean_recall,
            'mean_f1': mean_f1,
            'mean_iou': mean_iou,
            'class_metrics': class_metrics
        }
    
    def evaluate_dataset(self, data_loader, dataset_name="test"):
        """Evaluate model on a dataset."""
        print(f"Evaluating on {dataset_name} dataset...")
        
        all_predictions = []
        all_targets = []
        total_loss = 0.0
        num_batches = len(data_loader)
        
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(data_loader):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                outputs, loss = self.model(images, masks)
                
                if loss is not None:
                    total_loss += loss.item()
                
                # Get predictions
                if outputs.shape[1] > 1:
                    predictions = torch.argmax(outputs, dim=1)
                else:
                    predictions = (torch.sigmoid(outputs) > 0.5).long().squeeze(1)
                
                # Store predictions and targets
                all_predictions.append(predictions.cpu())
                all_targets.append(masks.cpu())
                
                if batch_idx % 10 == 0:
                    print(f"Processed {batch_idx}/{num_batches} batches")
        
        # Concatenate all predictions and targets
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Calculate metrics
        metrics = self.calculate_metrics(
            all_predictions, all_targets, 
            self.config.data_config["num_classes"]
        )
        
        metrics['average_loss'] = total_loss / num_batches
        
        return metrics, all_predictions, all_targets
    
    def plot_confusion_matrix(self, predictions, targets, dataset_name="test"):
        """Plot confusion matrix."""
        num_classes = self.config.data_config["num_classes"]
        
        # Flatten for confusion matrix
        pred_flat = predictions.flatten().numpy()
        target_flat = targets.flatten().numpy()
        
        # Calculate confusion matrix
        cm = confusion_matrix(target_flat, pred_flat, labels=range(num_classes))
        
        # Plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=[f'Class {i}' for i in range(num_classes)],
                   yticklabels=[f'Class {i}' for i in range(num_classes)])
        plt.title(f'Confusion Matrix - {dataset_name.title()} Set')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join(self.save_dir, f'confusion_matrix_{dataset_name}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return cm
    
    def visualize_predictions(self, data_loader, num_samples=5, dataset_name="test"):
        """Visualize model predictions."""
        self.model.eval()
        
        fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        sample_count = 0
        
        with torch.no_grad():
            for images, masks in data_loader:
                if sample_count >= num_samples:
                    break
                
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs, _ = self.model(images, masks)
                
                # Get predictions
                if outputs.shape[1] > 1:
                    predictions = torch.argmax(outputs, dim=1)
                else:
                    predictions = (torch.sigmoid(outputs) > 0.5).long().squeeze(1)
                
                # Process first image in batch
                image = images[0].cpu()
                mask = masks[0].cpu()
                prediction = predictions[0].cpu()
                
                # Denormalize image for visualization
                mean = torch.tensor(self.config.data_config["normalize_mean"]).view(3, 1, 1)
                std = torch.tensor(self.config.data_config["normalize_std"]).view(3, 1, 1)
                image = image * std + mean
                image = torch.clamp(image, 0, 1)
                
                # Plot
                axes[sample_count, 0].imshow(image.permute(1, 2, 0))
                axes[sample_count, 0].set_title('Input Image')
                axes[sample_count, 0].axis('off')
                
                axes[sample_count, 1].imshow(mask, cmap='tab10', vmin=0, vmax=self.config.data_config["num_classes"]-1)
                axes[sample_count, 1].set_title('Ground Truth')
                axes[sample_count, 1].axis('off')
                
                axes[sample_count, 2].imshow(prediction, cmap='tab10', vmin=0, vmax=self.config.data_config["num_classes"]-1)
                axes[sample_count, 2].set_title('Prediction')
                axes[sample_count, 2].axis('off')
                
                sample_count += 1
        
        plt.tight_layout()
        
        # Save visualization
        save_path = os.path.join(self.save_dir, f'predictions_visualization_{dataset_name}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_class_performance(self, metrics):
        """Analyze and visualize class-specific performance."""
        class_metrics = metrics['class_metrics']
        
        # Extract metrics for plotting
        classes = list(class_metrics.keys())
        precisions = [class_metrics[c]['precision'] for c in classes]
        recalls = [class_metrics[c]['recall'] for c in classes]
        f1_scores = [class_metrics[c]['f1'] for c in classes]
        ious = [class_metrics[c]['iou'] for c in classes]
        supports = [class_metrics[c]['support'] for c in classes]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Precision
        axes[0, 0].bar(classes, precisions, color='skyblue')
        axes[0, 0].set_title('Precision by Class')
        axes[0, 0].set_ylabel('Precision')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Recall
        axes[0, 1].bar(classes, recalls, color='lightcoral')
        axes[0, 1].set_title('Recall by Class')
        axes[0, 1].set_ylabel('Recall')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # F1 Score
        axes[1, 0].bar(classes, f1_scores, color='lightgreen')
        axes[1, 0].set_title('F1 Score by Class')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # IoU
        axes[1, 1].bar(classes, ious, color='gold')
        axes[1, 1].set_title('IoU by Class')
        axes[1, 1].set_ylabel('IoU')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join(self.save_dir, 'class_performance_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print class imbalance analysis
        print("\nClass Imbalance Analysis:")
        print("-" * 40)
        total_support = sum(supports)
        for i, (cls, support) in enumerate(zip(classes, supports)):
            percentage = (support / total_support) * 100
            print(f"{cls}: {support:,} samples ({percentage:.2f}%)")
            print(f"  Precision: {precisions[i]:.4f}")
            print(f"  Recall: {recalls[i]:.4f}")
            print(f"  F1: {f1_scores[i]:.4f}")
            print(f"  IoU: {ious[i]:.4f}")
            print()
    
    def generate_report(self, metrics, dataset_name="test"):
        """Generate comprehensive evaluation report."""
        report = {
            'dataset': dataset_name,
            'model_config': {
                'num_classes': self.config.data_config["num_classes"],
                'input_size': self.config.data_config["input_size"]
            },
            'overall_metrics': {
                'accuracy': metrics['overall_accuracy'],
                'mean_accuracy': metrics['mean_accuracy'],
                'mean_precision': metrics['mean_precision'],
                'mean_recall': metrics['mean_recall'],
                'mean_f1': metrics['mean_f1'],
                'mean_iou': metrics['mean_iou'],
                'average_loss': metrics.get('average_loss', 'N/A')
            },
            'class_metrics': metrics['class_metrics']
        }
        
        # Save report as JSON
        report_path = os.path.join(self.save_dir, f'evaluation_report_{dataset_name}.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print(f"\nEvaluation Report - {dataset_name.title()} Set")
        print("=" * 50)
        print(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}")
        print(f"Mean Class Accuracy: {metrics['mean_accuracy']:.4f}")
        print(f"Mean Precision: {metrics['mean_precision']:.4f}")
        print(f"Mean Recall: {metrics['mean_recall']:.4f}")
        print(f"Mean F1 Score: {metrics['mean_f1']:.4f}")
        print(f"Mean IoU: {metrics['mean_iou']:.4f}")
        if 'average_loss' in metrics:
            print(f"Average Loss: {metrics['average_loss']:.4f}")
        
        return report

def main():
    """Main evaluation function."""
    # Configuration
    config = get_config("experiment", "high_imbalance")
    model_path = "./aias_net_checkpoints/best_model.pth"  # Update this path
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please train the model first or update the model path.")
        return
    
    # Create data loaders
    data_loaders = get_data_loaders(config)
    
    # Initialize evaluator
    evaluator = AIASNetEvaluator(config, model_path, save_dir="./evaluation_results")
    
    # Evaluate on test set
    test_metrics, test_predictions, test_targets = evaluator.evaluate_dataset(
        data_loaders['test'], "test"
    )
    
    # Generate comprehensive report
    evaluator.generate_report(test_metrics, "test")
    
    # Visualizations
    evaluator.plot_confusion_matrix(test_predictions, test_targets, "test")
    evaluator.visualize_predictions(data_loaders['test'], num_samples=3, dataset_name="test")
    evaluator.analyze_class_performance(test_metrics)
    
    # Evaluate on validation set for comparison
    val_metrics, val_predictions, val_targets = evaluator.evaluate_dataset(
        data_loaders['val'], "validation"
    )
    evaluator.generate_report(val_metrics, "validation")
    
    print(f"\nEvaluation completed! Results saved to {evaluator.save_dir}")

if __name__ == "__main__":
    main()

