class Config:
    def __init__(self):
        # HRNet Backbone Configuration
        self.hrnet_config = {
            "input_channels": 3,
            "num_stages": 4,
            "num_channels_per_stage": [32, 64, 128, 256],
            "num_blocks_per_stage": [4, 4, 4, 4],
            "block_type": "BasicBlock"
        }
        
        # Dynamic Adaptive Sampling Configuration
        self.das_config = {
            "sampling_strategy": "adaptive",
            "threshold": 0.5,
            "min_samples_per_class": 10,
            "sampling_ratio": 0.8,  # Sample 80% of pixels
            "adaptive_learning": True
        }
        
        # Hybrid Imbalance-Aware Loss Function Configuration
        self.hialf_config = {
            "focal_loss_weight": 1.0,
            "dice_loss_weight": 1.0,
            "boundary_loss_weight": 0.5,
            "focal_alpha": 1.0,
            "focal_gamma": 2.0,
            "dice_smooth": 1e-6,
            "adaptive_weighting": True
        }
        
        # Meta-Learning Configuration
        self.meta_learning_config = {
            "learning_rate": 0.01,
            "num_meta_iterations": 5,
            "adaptation_lr": 0.001,
            "meta_batch_size": 4,
            "support_shots": 5,
            "query_shots": 15
        }
        
        # Training Configuration
        self.training_config = {
            "batch_size": 4,
            "learning_rate": 0.001,
            "num_epochs": 100,
            "optimizer": "Adam",
            "scheduler": "StepLR",
            "step_size": 30,
            "gamma": 0.1,
            "weight_decay": 1e-4
        }
        
        # Data Configuration
        self.data_config = {
            "input_size": (256, 256),
            "num_classes": 2,  # Binary segmentation by default
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
            "augmentation": {
                "horizontal_flip": True,
                "vertical_flip": True,
                "rotation": True,
                "color_jitter": True,
                "gaussian_blur": False
            }
        }
        
        # Evaluation Configuration
        self.eval_config = {
            "metrics": ["dice", "iou", "precision", "recall", "f1"],
            "save_predictions": True,
            "visualization": True,
            "class_specific_metrics": True
        }

class ExperimentConfig(Config):
    """Extended configuration for specific experiments."""
    
    def __init__(self, experiment_type="default"):
        super().__init__()
        
        if experiment_type == "high_imbalance":
            # Configuration for highly imbalanced datasets
            self.das_config["sampling_ratio"] = 0.9
            self.hialf_config["focal_gamma"] = 3.0
            self.hialf_config["boundary_loss_weight"] = 1.0
            
        elif experiment_type == "small_objects":
            # Configuration optimized for small object detection
            self.hrnet_config["num_channels_per_stage"] = [64, 128, 256, 512]
            self.hialf_config["boundary_loss_weight"] = 2.0
            self.data_config["input_size"] = (512, 512)
            
        elif experiment_type == "multi_class":
            # Configuration for multi-class segmentation
            self.data_config["num_classes"] = 5
            self.hialf_config["focal_alpha"] = 0.25
            self.das_config["min_samples_per_class"] = 20
            
        elif experiment_type == "fast_training":
            # Configuration for faster training
            self.training_config["batch_size"] = 8
            self.training_config["learning_rate"] = 0.002
            self.meta_learning_config["num_meta_iterations"] = 3
            self.data_config["input_size"] = (128, 128)

def get_config(config_type="default", experiment_type="default"):
    """
    Factory function to get configuration objects.
    
    Args:
        config_type: Type of configuration ("default" or "experiment")
        experiment_type: Type of experiment (for ExperimentConfig)
    
    Returns:
        Configuration object
    """
    if config_type == "experiment":
        return ExperimentConfig(experiment_type)
    else:
        return Config()

