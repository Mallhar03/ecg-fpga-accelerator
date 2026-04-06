import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

# Logger setup
logger = logging.getLogger("ecg_fpga.training.loss")

class MorphologyWeightedBCELoss(nn.Module):
    """
    Binary cross-entropy loss with per-class morphology weights.

    Assigns higher training penalty to misclassification of clinically
    critical rare signal classes (ST-segment, QT interval) versus
    common classes (Normal). Weights loaded from config — never hardcoded.

    Args:
        class_weights (list[float]): Per-class weights of shape (5,).
            Order: [ST_segment, QT_interval, P_wave, BBB, Normal].
        device (torch.device): Device to place the weight tensor on.

    Raises:
        ValueError: If class_weights does not have exactly 5 elements.
        ValueError: If any weight is <= 0.
    """
    def __init__(self, class_weights, device=None):
        super().__init__()
        if len(class_weights) != 5:
            error_msg = f"class_weights must have exactly 5 elements, got {len(class_weights)}"
            raise ValueError(error_msg)
        
        if any(w <= 0 for w in class_weights):
            error_msg = f"All class_weights must be > 0, got {class_weights}"
            raise ValueError(error_msg)
            
        self.register_buffer('class_weights', torch.tensor(class_weights, dtype=torch.float32, device=device))
        logger.debug(f"MorphologyWeightedBCELoss initialized with weights: {class_weights}")

    def forward(self, logits, targets):
        """
        Computes weighted binary cross entropy loss.

        Args:
            logits: (batch_size, 5) - raw model output (no sigmoid applied)
            targets: (batch_size, 5) - ground truth labels (0.0 or 1.0)

        Returns:
            torch.Tensor: Scalar loss value.
        """
        if logits.shape != targets.shape:
            error_msg = f"logits and targets must have matching shapes, got {logits.shape} vs {targets.shape}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # BCEWithLogitsLoss internally applies sigmoid
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.class_weights)
