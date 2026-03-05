# utils/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .metrics import compute_pos_weight

class CurriculumLoss(nn.Module):
    """
    Curriculum Loss that transitions from BCE to Focal Loss.
    
    This helps prevent the model from getting stuck in local minima
    when using focal loss on extremely imbalanced datasets.
    
    Args:
        alpha (float): Weight for positive class in focal loss (default: 0.25)
        gamma (float): Focusing parameter for focal loss (default: 2.0)
        transition_epochs (int): Number of epochs for transition (default: 10)
        warmup_epochs (int): Number of epochs to use only BCE (default: 5)
        eps (float): Small epsilon for numerical stability (default: 1e-6)
        pos_weight (torch.Tensor): Positive weight for BCE loss (default: None)
    """
    def __init__(self, alpha=0.25, gamma=2.0, transition_epochs=10, warmup_epochs=5, eps=1e-6, pos_weight=None):
        super(CurriculumLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.transition_epochs = transition_epochs
        self.warmup_epochs = warmup_epochs
        self.eps = eps
        self.current_epoch = 0
        self.pos_weight = pos_weight
        
    def set_epoch(self, epoch):
        """Set the current epoch for curriculum scheduling."""
        self.current_epoch = epoch
        
    def forward(self, inputs, targets):
        """
        Forward pass with curriculum learning.
        
        Args:
            inputs (torch.Tensor): Raw logits of shape (N,) or (N, 1)
            targets (torch.Tensor): Binary targets of shape (N,) or (N, 1)
            
        Returns:
            torch.Tensor: Computed curriculum loss
        """
        # Flatten inputs and targets
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Move pos_weight to the same device as inputs if it exists
        if self.pos_weight is not None:
            pos_weight = self.pos_weight.to(inputs.device)
            bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, pos_weight=pos_weight, reduction='none')
        else:
            bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Compute focal loss
        probs = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, probs, 1 - probs)
        pt = torch.clamp(pt, min=self.eps, max=1.0 - self.eps)
        
        focal_weight = (1 - pt) ** self.gamma
        alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        ce_loss = -torch.log(pt)
        focal_loss = alpha_weight * focal_weight * ce_loss
        
        # Determine curriculum weight
        if self.current_epoch < self.warmup_epochs:
            # Phase 1: BCE only
            curriculum_weight = 0.0
        elif self.current_epoch < self.warmup_epochs + self.transition_epochs:
            # Phase 2: Gradual transition
            progress = (self.current_epoch - self.warmup_epochs) / self.transition_epochs
            curriculum_weight = progress * 0.3  # Max 30% focal loss to prevent getting stuck
        else:
            # Phase 3: Blend (never pure focal loss)
            curriculum_weight = 0.3  # Keep 70% BCE, 30% focal loss
        
        # Combine losses
        final_loss = (1 - curriculum_weight) * bce_loss + curriculum_weight * focal_loss
        
        return final_loss.mean()

class FocalLoss(nn.Module):
    """
    Robust Focal Loss implementation for binary classification.
    
    Focal Loss addresses class imbalance by down-weighting easy examples and focusing on hard ones.
    FL(pt) = -αt * (1 - pt)^γ * log(pt)
    
    Args:
        alpha (float or torch.Tensor): Weight for positive class. 
            - float: Single weight applied to positive class
            - torch.Tensor: Per-class weights [weight_neg, weight_pos]
            Default: 0.25 (recommended for imbalanced datasets)
        gamma (float): Focusing parameter. Higher values focus more on hard examples.
            - gamma=0: Equivalent to BCE loss
            - gamma=1: Linear down-weighting of easy examples  
            - gamma=2: Standard focal loss (recommended)
            - gamma>2: More aggressive focusing
        reduction (str): Reduction method ('mean', 'sum', 'none')
        eps (float): Small epsilon for numerical stability (default: 1e-6)
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', eps=1e-6):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps
        
        # Validate alpha parameter
        if isinstance(alpha, torch.Tensor):
            if alpha.numel() != 2:
                raise ValueError("Alpha tensor must have 2 elements [weight_neg, weight_pos]")
        elif not isinstance(alpha, (int, float)):
            raise ValueError("Alpha must be float or torch.Tensor")
        
    def forward(self, inputs, targets):
        """
        Forward pass of focal loss.
        
        Args:
            inputs (torch.Tensor): Raw logits of shape (N,) or (N, 1)
            targets (torch.Tensor): Binary targets of shape (N,) or (N, 1)
            
        Returns:
            torch.Tensor: Computed focal loss
        """
        # Flatten inputs and targets to avoid shape issues
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Convert logits to probabilities with numerical stability
        probs = torch.sigmoid(inputs)
        
        # Calculate pt (probability for correct class) with clipping to avoid log(0)
        pt = torch.where(targets == 1, probs, 1 - probs)
        pt = torch.clamp(pt, min=self.eps, max=1.0 - self.eps)
        
        # Calculate focal weight: (1 - pt)^gamma
        # Remove problematic clamping that masks gradient signals
        focal_weight = (1 - pt) ** self.gamma
        
        # Handle alpha parameter safely
        if isinstance(self.alpha, torch.Tensor):
            # Per-class weighting: alpha[0] for negative, alpha[1] for positive
            alpha_weight = torch.where(targets == 1, self.alpha[1], self.alpha[0])
        else:
            # Single alpha value: apply to positive class, 1-alpha to negative
            # This ensures both classes get non-zero weights
            alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        # Calculate cross-entropy loss manually for better control
        ce_loss = -torch.log(pt)
        
        # Apply focal and alpha weights
        focal_loss = alpha_weight * focal_weight * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class AsymmetricFocalLoss(nn.Module):
    """
    Asymmetric Focal Loss with different gamma for positive/negative classes.
    
    Useful for flare forecasting where false positives and false negatives have different costs.
    This allows more aggressive focusing on the positive class (flares) while being more
    conservative with the negative class (no flares).
    
    Args:
        alpha (float or torch.Tensor): Weight for positive class (default: 0.25)
        gamma_pos (float): Focusing parameter for positive class (default: 2.0)
        gamma_neg (float): Focusing parameter for negative class (default: 1.0)
        reduction (str): Reduction method ('mean', 'sum', 'none')
        eps (float): Small epsilon for numerical stability (default: 1e-6)
    """
    def __init__(self, alpha=0.25, gamma_pos=2.0, gamma_neg=1.0, reduction='mean', eps=1e-6):
        super(AsymmetricFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.reduction = reduction
        self.eps = eps
        
        # Validate alpha parameter
        if isinstance(alpha, torch.Tensor):
            if alpha.numel() != 2:
                raise ValueError("Alpha tensor must have 2 elements [weight_neg, weight_pos]")
        elif not isinstance(alpha, (int, float)):
            raise ValueError("Alpha must be float or torch.Tensor")
        
    def forward(self, inputs, targets):
        """
        Forward pass of asymmetric focal loss.
        
        Args:
            inputs (torch.Tensor): Raw logits of shape (N,) or (N, 1)
            targets (torch.Tensor): Binary targets of shape (N,) or (N, 1)
            
        Returns:
            torch.Tensor: Computed asymmetric focal loss
        """
        # Flatten inputs and targets to avoid shape issues
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Convert logits to probabilities with numerical stability
        probs = torch.sigmoid(inputs)
        
        # Calculate pt (probability for correct class) with clipping to avoid log(0)
        pt = torch.where(targets == 1, probs, 1 - probs)
        pt = torch.clamp(pt, min=self.eps, max=1.0 - self.eps)
        
        # Different focusing for positive and negative classes
        gamma = torch.where(targets == 1, self.gamma_pos, self.gamma_neg)
        
        # Calculate focal weight: (1 - pt)^gamma (no problematic clamping)
        focal_weight = (1 - pt) ** gamma
        
        # Handle alpha parameter safely
        if isinstance(self.alpha, torch.Tensor):
            # Per-class weighting: alpha[0] for negative, alpha[1] for positive
            alpha_weight = torch.where(targets == 1, self.alpha[1], self.alpha[0])
        else:
            # Single alpha value: apply to positive class, 1-alpha to negative
            alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        # Calculate cross-entropy loss manually for better control
        ce_loss = -torch.log(pt)
        
        # Apply focal and alpha weights
        focal_loss = alpha_weight * focal_weight * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def create_loss_function(loss_config, dataset=None, device="cpu"):
    """
    Create loss function based on configuration.
    
    Args:
        loss_config: Dictionary containing loss function configuration
        dataset: Dataset for computing positive weight if needed
        device: Device to create tensors on (default: "cpu")
    
    Returns:
        Loss function instance
    """
    loss_type = loss_config.get("type", "bce")
    
    if loss_type == "bce":
        # Standard BCE with positive weight
        pos_weight = compute_pos_weight(dataset) if dataset else torch.tensor([1.0])
        pos_weight = pos_weight.to(device)
        return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    elif loss_type == "curriculum":
        # Curriculum loss (BCE → Focal)
        curriculum_config = loss_config.get("curriculum", {})
        alpha = curriculum_config.get("alpha", 0.25)
        gamma = curriculum_config.get("gamma", 2.0)
        warmup_epochs = curriculum_config.get("warmup_epochs", 5)
        transition_epochs = curriculum_config.get("transition_epochs", 10)
        eps = curriculum_config.get("eps", 1e-6)
        pos_weight = compute_pos_weight(dataset) if dataset else torch.tensor([1.0])
        pos_weight = pos_weight.to(device)
        print(f"✅ Using Curriculum Loss (BCE → Focal, alpha={alpha}, gamma={gamma}, warmup={warmup_epochs}, transition={transition_epochs}, pos_weight={pos_weight.item():.2f})")
        return CurriculumLoss(alpha=alpha, gamma=gamma, warmup_epochs=warmup_epochs, transition_epochs=transition_epochs, eps=eps, pos_weight=pos_weight)
    
    elif loss_type == "focal":
        # Standard focal loss
        focal_config = loss_config.get("focal", {})
        alpha = focal_config.get("alpha", 0.25)  # Better default for imbalanced data
        gamma = focal_config.get("gamma", 2.0)
        eps = focal_config.get("eps", 1e-6)
        print(f"✅ Using Focal Loss (alpha={alpha}, gamma={gamma}, eps={eps})")
        return FocalLoss(alpha=alpha, gamma=gamma, eps=eps)
    
    elif loss_type == "asymmetric_focal":
        # Asymmetric focal loss with different gamma for pos/neg
        asym_config = loss_config.get("asymmetric_focal", {})
        alpha = asym_config.get("alpha", 0.25)  # Better default for imbalanced data
        gamma_pos = asym_config.get("gamma_pos", 2.0)
        gamma_neg = asym_config.get("gamma_neg", 1.0)
        eps = asym_config.get("eps", 1e-6)
        print(f"✅ Using Asymmetric Focal Loss (alpha={alpha}, gamma_pos={gamma_pos}, gamma_neg={gamma_neg}, eps={eps})")
        return AsymmetricFocalLoss(alpha=alpha, gamma_pos=gamma_pos, gamma_neg=gamma_neg, eps=eps)
    
    else:
        print(f"⚠️ Unknown loss type '{loss_type}', falling back to BCE")
        pos_weight = compute_pos_weight(dataset) if dataset else torch.tensor([1.0])
        pos_weight = pos_weight.to(device)
        return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight) 