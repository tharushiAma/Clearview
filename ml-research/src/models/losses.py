"""
Loss functions for handling class imbalance in multi-aspect sentiment analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    
    Args:
        alpha: Weighting factor for each class
        gamma: Focusing parameter (higher = more focus on hard examples)
        reduction: 'mean', 'sum', or 'none'
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (batch_size, num_classes) logits
            targets: (batch_size,) class labels
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)          # pt = probability assigned to the correct class
        focal_loss = (1 - pt) ** self.gamma * ce_loss  # Down-weights easy examples (high pt)
        
        if self.alpha is not None:
            # alpha can be passed as a Python list, numpy array, torch Tensor, or a scalar.
            # In all cases we index by target class to get per-sample weights (alpha_t).
            if isinstance(self.alpha, (list, np.ndarray)):
                alpha_t = torch.tensor(self.alpha, device=inputs.device, dtype=torch.float)[targets]
            elif isinstance(self.alpha, torch.Tensor):
                alpha_t = self.alpha.to(inputs.device)[targets]
            else:
                # Scalar alpha — same weight applied to every sample
                alpha_t = self.alpha
            focal_loss = alpha_t * focal_loss  # Scale each sample's loss by its class weight
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ClassBalancedLoss(nn.Module):
    """
    Class-Balanced Loss based on Effective Number of Samples
    
    Works better than standard weighted loss for extreme imbalance
    
    Args:
        samples_per_class: List of sample counts for each class
        beta: Hyperparameter (0.9999 for extreme imbalance, 0.999 for moderate)
        reduction: 'mean', 'sum', or 'none'
    """
    def __init__(self, samples_per_class, beta=0.9999, reduction='mean'):
        super(ClassBalancedLoss, self).__init__()
        # Effective Number formula: EN(n) = (1 - beta^n) / (1 - beta)
        # As n grows, EN grows logarithmically rather than linearly.
        # This gives more meaningful weights for very large majority classes
        # compared to simple inverse-frequency weighting.
        effective_num = 1.0 - np.power(beta, samples_per_class)
        weights = (1.0 - beta) / np.array(effective_num)
        # Re-normalise so weights average to 1.0 (preserves the loss scale)
        weights = weights / weights.sum() * len(weights)
        self.weights = torch.tensor(weights, dtype=torch.float32)
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (batch_size, num_classes) logits
            targets: (batch_size,) class labels
        """
        # Move weights to the same device as inputs at runtime in case the model
        # is on GPU but the weights tensor was created on CPU.
        if self.weights.device != inputs.device:
            self.weights = self.weights.to(inputs.device)
        return F.cross_entropy(inputs, targets, weight=self.weights, reduction=self.reduction)





class DiceLoss(nn.Module):
    """
    Dice Loss for imbalanced classification
    Directly optimizes F1-score (Dice coefficient)
    
    Args:
        smooth: Smoothing factor to avoid division by zero
    """
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (batch_size, num_classes) logits
            targets: (batch_size,) class labels
        """
        inputs = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()
        
        intersection = (inputs * targets_one_hot).sum(dim=0)
        cardinality = inputs.sum(dim=0) + targets_one_hot.sum(dim=0)
        
        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice_score.mean()


class HybridLoss(nn.Module):
    """
    Combination of multiple losses for extreme class imbalance
    Recommended for severely imbalanced aspects like price and packing
    
    Args:
        samples_per_class: List of sample counts [neg, neu, pos]
        focal_alpha: Alpha parameter for focal loss (per-class weights)
        focal_gamma: Gamma parameter for focal loss (focusing parameter)
        cb_beta: Beta parameter for class-balanced loss
        weights: Dictionary with weights for each loss component
    """
    def __init__(self, samples_per_class, focal_alpha=None, focal_gamma=2.0, 
                 cb_beta=0.9999, weights=None):
        super(HybridLoss, self).__init__()
        
        if weights is None:
            # Default: Focal is the primary term (weight=1.0), CB corrects for
            # extreme imbalance (weight=0.5), Dice improves minority-class F1 (weight=0.3).
            # The A7 ablation found dice=0.0 was optimal for this dataset, so
            # config.yaml overrides this default via loss_weights.
            weights = {'focal': 1.0, 'cb': 0.5, 'dice': 0.3}
        
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.cb_loss = ClassBalancedLoss(samples_per_class, beta=cb_beta)
        self.dice_loss = DiceLoss()
        self.weights = weights
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (batch_size, num_classes) logits
            targets: (batch_size,) class labels
            
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with individual loss values
        """
        loss_focal = self.focal_loss(inputs, targets)
        loss_cb = self.cb_loss(inputs, targets)
        loss_dice = self.dice_loss(inputs, targets)
        
        total_loss = (self.weights.get('focal', 0.0) * loss_focal + 
                     self.weights.get('cb', 0.0) * loss_cb + 
                     self.weights.get('dice', 0.0) * loss_dice)
        
        loss_dict = {
            'focal': loss_focal.item(),
            'cb': loss_cb.item(),
            'dice': loss_dice.item(),
            'total': total_loss.item()
        }
        
        return total_loss, loss_dict


class AspectSpecificLossManager:
    """
    Manages aspect-specific loss functions based on class distribution
    Automatically selects appropriate loss parameters for each aspect
    """
    def __init__(self, aspect_class_counts, config):
        """
        Args:
            aspect_class_counts: Dict mapping aspect names to [neg, neu, pos] counts
            config: Configuration dictionary with loss parameters
        """
        self.aspect_losses = {}
        self.config = config
        
        for aspect, counts in aspect_class_counts.items():
            # Calculate imbalance ratio
            max_count = max(counts)
            min_count = min(counts)
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
            
            # Gamma controls how strongly the loss focuses on hard (misclassified) examples.
            # Higher gamma = more aggressive down-weighting of easy examples.
            if aspect in config.get('focal_gamma', {}):
                gamma = config['focal_gamma'][aspect]  # Per-aspect override from config
            elif imbalance_ratio > 50:  # Severe imbalance (e.g. price: 2244 pos vs 17 neg)
                gamma = 3.0
            elif imbalance_ratio > 10:  # Moderate imbalance
                gamma = 2.5
            else:
                gamma = 2.0  # Standard focal loss default
            
            # Beta controls the effective-number calculation in CB Loss.
            # 0.9999 recommended for extreme imbalance (Cui et al. 2019).
            if aspect in config.get('class_balanced_beta', {}):
                beta = config['class_balanced_beta'][aspect]  # Per-aspect override
            elif imbalance_ratio > 50:
                beta = 0.9999
            else:
                beta = 0.999
            
            # Inverse-frequency alpha: rare classes get proportionally larger weights.
            # Clamped at 1.0 to avoid division by zero for empty classes.
            total = sum(counts)
            focal_alpha = [total / (len(counts) * c) if c > 0 else 1.0 for c in counts]
            focal_alpha = torch.tensor(focal_alpha, dtype=torch.float32)
            
            # Create hybrid loss
            self.aspect_losses[aspect] = HybridLoss(
                samples_per_class=counts,
                focal_alpha=focal_alpha,
                focal_gamma=gamma,
                cb_beta=beta,
                weights=config.get('loss_weights', {'focal': 1.0, 'cb': 0.5, 'dice': 0.3})
            )
            
            print(f"Initialized loss for {aspect}:")
            print(f"  Class counts: {counts}")
            print(f"  Imbalance ratio: {imbalance_ratio:.2f}")
            print(f"  Focal gamma: {gamma}")
            print(f"  CB beta: {beta}")
            print(f"  Focal alpha: {focal_alpha.tolist()}")
    
    def get_loss(self, aspect):
        """Get loss function for specific aspect"""
        return self.aspect_losses[aspect]
    
    def compute_loss(self, predictions, targets, aspect_ids, aspect_names):
        """
        Compute loss for a batch with multiple aspects
        
        Args:
            predictions: (batch_size, num_classes) logits
            targets: (batch_size,) class labels
            aspect_ids: (batch_size,) aspect indices
            aspect_names: List of aspect names
            
        Returns:
            total_loss: Average loss across batch
            loss_details: Dictionary with per-aspect loss breakdown
        """
        total_loss = 0
        loss_details = {}
        
        for i in range(predictions.size(0)):
            aspect_idx = aspect_ids[i].item()
            aspect = aspect_names[aspect_idx]
            
            loss_fn = self.aspect_losses[aspect]
            sample_loss, loss_dict = loss_fn(
                predictions[i].unsqueeze(0),
                targets[i].unsqueeze(0)
            )
            
            total_loss += sample_loss
            
            # Accumulate loss details
            for key, value in loss_dict.items():
                if aspect not in loss_details:
                    loss_details[aspect] = {}
                if key not in loss_details[aspect]:
                    loss_details[aspect][key] = []
                loss_details[aspect][key].append(value)
        
        total_loss = total_loss / predictions.size(0)
        
        # Average loss details
        for aspect in loss_details:
            for key in loss_details[aspect]:
                loss_details[aspect][key] = np.mean(loss_details[aspect][key])
        
        return total_loss, loss_details


if __name__ == "__main__":
    # Test the loss functions
    print("Testing loss functions...")
    
    # Simulate severely imbalanced data (like price aspect)
    samples_per_class = [17, 15, 2244]  # neg, neu, pos
    batch_size = 8
    num_classes = 3
    
    # Create dummy predictions and targets
    predictions = torch.randn(batch_size, num_classes)
    targets = torch.tensor([2, 2, 2, 2, 0, 2, 1, 2])  # Mostly positive class
    
    # Test Focal Loss
    focal_loss = FocalLoss(gamma=3.0)
    loss = focal_loss(predictions, targets)
    print(f"\nFocal Loss: {loss.item():.4f}")
    
    # Test Class-Balanced Loss
    cb_loss = ClassBalancedLoss(samples_per_class, beta=0.9999)
    loss = cb_loss(predictions, targets)
    print(f"Class-Balanced Loss: {loss.item():.4f}")
    
    # Test Dice Loss
    dice_loss = DiceLoss()
    loss = dice_loss(predictions, targets)
    print(f"Dice Loss: {loss.item():.4f}")
    
    # Test Hybrid Loss
    hybrid_loss = HybridLoss(samples_per_class, focal_gamma=3.0)
    loss, loss_dict = hybrid_loss(predictions, targets)
    print(f"\nHybrid Loss: {loss.item():.4f}")
    print(f"Loss breakdown: {loss_dict}")
    
    print("\nAll tests passed!")
