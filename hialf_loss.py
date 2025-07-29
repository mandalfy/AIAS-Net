import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # Apply softmax to inputs if they are logits
        if inputs.dim() > 1 and inputs.shape[1] > 1:
            inputs = F.softmax(inputs, dim=1)
        else:
            inputs = torch.sigmoid(inputs)
        
        # Flatten tensors
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1).float()
        
        # Calculate intersection and union
        intersection = (inputs_flat * targets_flat).sum()
        dice_coeff = (2. * intersection + self.smooth) / (inputs_flat.sum() + targets_flat.sum() + self.smooth)
        
        return 1 - dice_coeff

class BoundaryAwareLoss(nn.Module):
    def __init__(self, boundary_weight=1.0):
        super(BoundaryAwareLoss, self).__init__()
        self.boundary_weight = boundary_weight
        
        # Sobel kernels for edge detection
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    def forward(self, inputs, targets):
        device = inputs.device
        self.sobel_x = self.sobel_x.to(device)
        self.sobel_y = self.sobel_y.to(device)
        
        # Apply softmax to inputs if they are logits
        if inputs.dim() > 1 and inputs.shape[1] > 1:
            inputs = F.softmax(inputs, dim=1)
        else:
            inputs = torch.sigmoid(inputs)
        
        # Convert targets to float
        targets = targets.float()
        
        # Ensure inputs and targets have the same shape
        if inputs.dim() == 4 and targets.dim() == 3:
            targets = targets.unsqueeze(1)
        elif inputs.dim() == 3 and targets.dim() == 3:
            inputs = inputs.unsqueeze(1)
            targets = targets.unsqueeze(1)
        
        # Calculate gradients (edges) for both predictions and targets
        pred_grad_x = F.conv2d(inputs, self.sobel_x, padding=1)
        pred_grad_y = F.conv2d(inputs, self.sobel_y, padding=1)
        pred_edges = torch.sqrt(pred_grad_x**2 + pred_grad_y**2)
        
        target_grad_x = F.conv2d(targets, self.sobel_x, padding=1)
        target_grad_y = F.conv2d(targets, self.sobel_y, padding=1)
        target_edges = torch.sqrt(target_grad_x**2 + target_grad_y**2)
        
        # Calculate boundary loss as MSE between edge maps
        boundary_loss = F.mse_loss(pred_edges, target_edges)
        
        return self.boundary_weight * boundary_loss

class HybridImbalanceAwareLoss(nn.Module):
    def __init__(self, config):
        super(HybridImbalanceAwareLoss, self).__init__()
        self.focal_loss_weight = config.get("focal_loss_weight", 1.0)
        self.dice_loss_weight = config.get("dice_loss_weight", 1.0)
        self.boundary_loss_weight = config.get("boundary_loss_weight", 0.5)
        
        # Initialize individual loss components
        self.focal_loss = FocalLoss(alpha=config.get("focal_alpha", 1), gamma=config.get("focal_gamma", 2))
        self.dice_loss = DiceLoss(smooth=config.get("dice_smooth", 1e-6))
        self.boundary_loss = BoundaryAwareLoss(boundary_weight=1.0)
        
        # Learnable adaptive weights
        self.adaptive_weights = nn.Parameter(torch.tensor([self.focal_loss_weight, self.dice_loss_weight, self.boundary_loss_weight]))
        
    def forward(self, outputs, targets):
        # Ensure targets are in the correct format
        if targets.dtype != torch.long and outputs.shape[1] > 1:
            targets = targets.long()
        
        # Calculate individual loss components
        focal_loss_val = self.focal_loss(outputs, targets)
        dice_loss_val = self.dice_loss(outputs, targets)
        boundary_loss_val = self.boundary_loss(outputs, targets)
        
        # Apply adaptive weighting (softmax to ensure weights sum to reasonable values)
        weights = F.softmax(self.adaptive_weights, dim=0)
        
        # Combine losses with adaptive weights
        total_loss = (weights[0] * focal_loss_val + 
                     weights[1] * dice_loss_val + 
                     weights[2] * boundary_loss_val)
        
        return total_loss
    
    def get_individual_losses(self, outputs, targets):
        """Return individual loss components for monitoring."""
        focal_loss_val = self.focal_loss(outputs, targets)
        dice_loss_val = self.dice_loss(outputs, targets)
        boundary_loss_val = self.boundary_loss(outputs, targets)
        
        return {
            'focal_loss': focal_loss_val.item(),
            'dice_loss': dice_loss_val.item(),
            'boundary_loss': boundary_loss_val.item(),
            'adaptive_weights': F.softmax(self.adaptive_weights, dim=0).detach().cpu().numpy()
        }

