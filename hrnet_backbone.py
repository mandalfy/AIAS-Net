import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class HRNetBackbone(nn.Module):
    def __init__(self, config):
        super(HRNetBackbone, self).__init__()
        self.input_channels = config["input_channels"]
        self.num_stages = config["num_stages"]
        self.num_channels_per_stage = config["num_channels_per_stage"]
        
        # Initial convolution layers
        self.conv1 = nn.Conv2d(self.input_channels, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Build stages
        self.stages = nn.ModuleList()
        for i in range(self.num_stages):
            stage = self._make_stage(i)
            self.stages.append(stage)
        
        # Fusion layers for multi-scale feature fusion
        self.fusion_layers = nn.ModuleList()
        for i in range(self.num_stages - 1):
            fusion_layer = self._make_fusion_layer(i)
            self.fusion_layers.append(fusion_layer)

    def _make_stage(self, stage_idx):
        # Simplified stage creation - in practice, this would be more complex
        if stage_idx == 0:
            in_channels = 64
        else:
            in_channels = self.num_channels_per_stage[stage_idx - 1]
        
        out_channels = self.num_channels_per_stage[stage_idx]
        
        layers = []
        layers.append(BasicBlock(in_channels, out_channels))
        for _ in range(3):  # Add more blocks for each stage
            layers.append(BasicBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)

    def _make_fusion_layer(self, stage_idx):
        # Simplified fusion layer - combines features from different resolution streams
        in_channels = self.num_channels_per_stage[stage_idx]
        out_channels = self.num_channels_per_stage[stage_idx + 1]
        
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Initial convolutions
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        # Process through stages with multi-resolution streams
        features = [x]
        
        for i, stage in enumerate(self.stages):
            x = stage(x)
            features.append(x)
            
            # Apply fusion if not the last stage
            if i < len(self.fusion_layers):
                x = self.fusion_layers[i](x)
        
        # Aggregate multi-scale features (simplified)
        # In practice, this would involve more sophisticated attention mechanisms
        aggregated_features = features[-1]  # Use the last stage features for now
        
        return aggregated_features

