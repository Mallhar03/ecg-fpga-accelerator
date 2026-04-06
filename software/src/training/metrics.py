import numpy as np
import logging

logger = logging.getLogger("ecg_fpga.training.metrics")

CLASS_NAMES = ["ST_segment", "QT_interval", "P_wave", "Bundle_Branch_Block", "Normal"]

def compute_clinical_metrics(y_true: np.ndarray, y_pred_probs: np.ndarray,
                              threshold: float = 0.5) -> dict:
    """
    Compute per-class and macro-average clinical metrics.

    Args:
        y_true (np.ndarray): Ground truth labels, shape (N, 5), values 0 or 1.
        y_pred_probs (np.ndarray): Predicted probabilities after sigmoid,
            shape (N, 5), values in [0, 1].
        threshold (float): Decision threshold for binary classification.

    Returns:
        dict: Keys are class names + 'macro_avg'. Each value is a dict with:
            'sensitivity' (float): TP / (TP + FN). Also called recall.
            'specificity' (float): TN / (TN + FP).
            'f1' (float): 2 * precision * recall / (precision + recall).
        Raises:
            ValueError: If y_true or y_pred_probs shape is not (N, 5).
            ValueError: If shapes don't match.
    """
    if y_true.shape != y_pred_probs.shape:
        error_msg = f"Shapes don't match: y_true {y_true.shape}, y_pred_probs {y_pred_probs.shape}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    if len(y_true.shape) != 2 or y_true.shape[1] != 5:
        error_msg = f"Expected shape (N, 5), got {y_true.shape}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    y_pred = (y_pred_probs >= threshold).astype(int)
    
    metrics = {}
    
    sensitivities = []
    specificities = []
    f1s = []
    
    for i, class_name in enumerate(CLASS_NAMES):
        tp = np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 1))
        tn = np.sum((y_true[:, i] == 0) & (y_pred[:, i] == 0))
        fp = np.sum((y_true[:, i] == 0) & (y_pred[:, i] == 1))
        fn = np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 0))
        
        # Sensitivity (Recall)
        if (tp + fn) == 0:
            logger.warning(f"Class {class_name} has zero positive samples. Setting sensitivity to 0.0")
            sensitivity = 0.0
        else:
            sensitivity = tp / (tp + fn)
            
        # Specificity
        if (tn + fp) == 0:
            logger.warning(f"Class {class_name} has zero negative samples. Setting specificity to 0.0")
            specificity = 0.0
        else:
            specificity = tn / (tn + fp)
            
        # F1 Score
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = sensitivity
        if (precision + recall) == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
            
        metrics[class_name] = {
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'f1': float(f1)
        }
        
        sensitivities.append(sensitivity)
        specificities.append(specificity)
        f1s.append(f1)
        
    metrics['macro_avg'] = {
        'sensitivity': float(np.mean(sensitivities)),
        'specificity': float(np.mean(specificities)),
        'f1': float(np.mean(f1s))
    }
    
    return metrics
