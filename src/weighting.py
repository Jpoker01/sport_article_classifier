"""Class weighting for imbalanced training.
 
Provides softened and capped inverse-frequency class weights, in the two
shapes needed by this project:  
- a numpy array (for torch's cross-entropy in the transformer trainer) 
- {label: weight} dict (for sklearn's`class_weight` parameter). 

"""
import numpy as np
 
from src.config import CLASS_WEIGHT_BETA, CLASS_WEIGHT_CAP
 
 
def compute_capped_class_weight(y, num_labels,
                                beta=CLASS_WEIGHT_BETA, cap=CLASS_WEIGHT_CAP):
    """Compute class weights: bigger for rare classes, smaller for common ones,
    but with a ceiling so tiny classes cannot dominate the loss.

    `beta` controls how strongly rare classes are boosted (1.0 = sklearn's
    "balanced", 0.5 = sqrt-damped, 0.0 = uniform). `cap` is the maximum ratio
    between the largest and smallest weight. Result is normalized so the mean is 1.
 
    Args:
        y: Integer training labels.
        num_labels: Total number of classes
        beta: Exponent on the inverse-frequency weight, in [0.0, 1.0].
        cap: Maximum allowed ratio between the largest and smallest weight.
 
    Returns:
        1-D numpy array of length `num_labels` with mean 1.
    """
    counts = np.bincount(y, minlength=num_labels).astype(float)
    counts = np.maximum(counts, 1.0)  # avoid div-by-zero for absent classes
    weights = (counts.sum() / (num_labels * counts)) ** beta
    weights = np.minimum(weights, cap * weights.min())
    return weights / weights.mean()
 
 
def capped_class_weight_dict(y, num_labels,
                             beta=CLASS_WEIGHT_BETA, cap=CLASS_WEIGHT_CAP):
    """Compute class weights: bigger for rare classes, smaller for common ones,
    but with a ceiling so tiny classes cannot dominate the loss. Packaged as a
    {int_label: float_weight} dict for sklearn's `class_weight` parameter.

    `beta` controls how strongly rare classes are boosted (1.0 = sklearn's
    "balanced", 0.5 = sqrt-damped, 0.0 = uniform). `cap` is the maximum ratio
    between the largest and smallest weight. Result is normalized so the mean is 1. 
    """
    weights = compute_capped_class_weight(y, num_labels, beta, cap)
    return {int(i): float(w) for i, w in enumerate(weights)}