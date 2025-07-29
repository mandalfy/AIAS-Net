import torch
import torch.nn as nn

from .hrnet_backbone import HRNetBackbone
from .das_module import DynamicAdaptiveSampling
from .hialf_loss import HybridImbalanceAwareLoss
from .meta_learning import MetaLearningModule

class AIASNet(nn.Module):
    def __init__(self, num_classes, config):
        super(AIASNet, self).__init__()
        self.hrnet = HRNetBackbone(config.hrnet_config)
        self.das = DynamicAdaptiveSampling(config.das_config)
        self.hialf = HybridImbalanceAwareLoss(config.hialf_config)
        self.meta_learner = MetaLearningModule(config.meta_learning_config)
        self.segmentation_head = nn.Conv2d(config.hrnet_config["num_channels_per_stage"][-1], num_classes, kernel_size=1) # Example segmentation head

    def forward(self, x, targets=None):
        # HRNet Backbone forward pass
        features = self.hrnet(x)

        # Dynamic Adaptive Sampling (DAS) - applied during training
        if self.training and targets is not None:
            sampled_features, sampled_targets = self.das(features, targets)
        else:
            sampled_features = features # For inference, no sampling
            sampled_targets = targets

        # Segmentation head
        output = self.segmentation_head(sampled_features)

        loss = None
        if sampled_targets is not None:
            # Hybrid Imbalance-Aware Loss Function (HIALF)
            loss = self.hialf(output, sampled_targets)

        return output, loss


