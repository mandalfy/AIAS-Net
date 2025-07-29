import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict

class MetaLearningModule(nn.Module):
    def __init__(self, config):
        super(MetaLearningModule, self).__init__()
        self.learning_rate = config.get("learning_rate", 0.01)
        self.num_meta_iterations = config.get("num_meta_iterations", 5)
        self.adaptation_lr = config.get("adaptation_lr", 0.001)
        
        # Meta-network for learning adaptation strategies
        self.meta_network = nn.Sequential(
            nn.Linear(3, 64),  # Input: [focal_loss, dice_loss, boundary_loss]
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3)   # Output: adaptation weights for DAS and HIALF
        )
        
        # Optimizer for meta-learning
        self.meta_optimizer = optim.Adam(self.meta_network.parameters(), lr=self.learning_rate)
        
    def forward(self, loss_components, model_parameters=None):
        """
        Perform meta-learning adaptation based on loss components.
        
        Args:
            loss_components: Dictionary containing individual loss values
            model_parameters: Current model parameters (for future use)
        
        Returns:
            adaptation_weights: Weights for adapting DAS and HIALF parameters
        """
        # Extract loss values
        focal_loss = loss_components.get('focal_loss', 0.0)
        dice_loss = loss_components.get('dice_loss', 0.0)
        boundary_loss = loss_components.get('boundary_loss', 0.0)
        
        # Create input tensor for meta-network
        loss_input = torch.tensor([focal_loss, dice_loss, boundary_loss], dtype=torch.float32)
        if loss_input.device != next(self.meta_network.parameters()).device:
            loss_input = loss_input.to(next(self.meta_network.parameters()).device)
        
        # Forward pass through meta-network
        adaptation_weights = self.meta_network(loss_input)
        
        # Apply softmax to ensure weights are normalized
        adaptation_weights = torch.softmax(adaptation_weights, dim=0)
        
        return adaptation_weights
    
    def meta_update(self, support_losses, query_losses, das_module, hialf_module):
        """
        Perform meta-learning update using support and query sets.
        
        Args:
            support_losses: Loss components from support set
            query_losses: Loss components from query set
            das_module: Dynamic Adaptive Sampling module
            hialf_module: Hybrid Imbalance-Aware Loss Function module
        """
        self.meta_optimizer.zero_grad()
        
        # Get adaptation weights for support set
        support_weights = self.forward(support_losses)
        
        # Clone parameters for adaptation
        das_params = OrderedDict(das_module.named_parameters())
        hialf_params = OrderedDict(hialf_module.named_parameters())
        
        # Adapt parameters using support set
        adapted_das_params = self._adapt_parameters(das_params, support_weights[0])
        adapted_hialf_params = self._adapt_parameters(hialf_params, support_weights[1:])
        
        # Evaluate on query set with adapted parameters
        query_weights = self.forward(query_losses)
        
        # Calculate meta-loss (simplified - in practice, this would involve 
        # evaluating the adapted model on the query set)
        meta_loss = self._calculate_meta_loss(query_weights, support_weights)
        
        # Backward pass and update meta-network
        meta_loss.backward()
        self.meta_optimizer.step()
        
        return meta_loss.item()
    
    def _adapt_parameters(self, parameters, adaptation_weight):
        """
        Adapt parameters based on adaptation weight.
        
        Args:
            parameters: OrderedDict of parameters
            adaptation_weight: Weight for adaptation
        
        Returns:
            adapted_parameters: Adapted parameters
        """
        adapted_parameters = OrderedDict()
        
        for name, param in parameters.items():
            if param.requires_grad:
                # Simple adaptation: scale parameters by adaptation weight
                adapted_param = param * (1 + adaptation_weight * self.adaptation_lr)
                adapted_parameters[name] = adapted_param
            else:
                adapted_parameters[name] = param
        
        return adapted_parameters
    
    def _calculate_meta_loss(self, query_weights, support_weights):
        """
        Calculate meta-loss based on query and support weights.
        
        Args:
            query_weights: Adaptation weights for query set
            support_weights: Adaptation weights for support set
        
        Returns:
            meta_loss: Meta-learning loss
        """
        # Simple meta-loss: encourage consistency between support and query adaptations
        consistency_loss = torch.mean((query_weights - support_weights) ** 2)
        
        # Add regularization to prevent extreme adaptations
        regularization_loss = torch.mean(torch.abs(support_weights))
        
        meta_loss = consistency_loss + 0.01 * regularization_loss
        
        return meta_loss
    
    def get_adaptation_strategy(self, loss_history):
        """
        Get adaptation strategy based on loss history.
        
        Args:
            loss_history: List of loss component dictionaries
        
        Returns:
            strategy: Dictionary containing adaptation recommendations
        """
        if len(loss_history) < 2:
            return {"status": "insufficient_history"}
        
        # Analyze loss trends
        recent_losses = loss_history[-5:]  # Last 5 iterations
        
        focal_trend = [l.get('focal_loss', 0) for l in recent_losses]
        dice_trend = [l.get('dice_loss', 0) for l in recent_losses]
        boundary_trend = [l.get('boundary_loss', 0) for l in recent_losses]
        
        # Calculate trends (simplified)
        focal_increasing = focal_trend[-1] > focal_trend[0]
        dice_increasing = dice_trend[-1] > dice_trend[0]
        boundary_increasing = boundary_trend[-1] > boundary_trend[0]
        
        strategy = {
            "focal_adaptation": "increase_weight" if focal_increasing else "decrease_weight",
            "dice_adaptation": "increase_weight" if dice_increasing else "decrease_weight",
            "boundary_adaptation": "increase_weight" if boundary_increasing else "decrease_weight",
            "overall_trend": "improving" if not (focal_increasing and dice_increasing) else "degrading"
        }
        
        return strategy

