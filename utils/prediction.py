"""
SeedLab AI — Inference & Probability Distribution Engine.
Handles real-time digit recognition, calibrated probabilities, and multi-model evaluation.
"""

from typing import Dict, Any, List, Optional
import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    """Compute numerically stable softmax values for logits."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def predict_digit_probabilities(estimator: Any, flat_vector: np.ndarray) -> Dict[str, Any]:
    """
    Infers the digit class and produces a calibrated 10-class probability distribution.

    Args:
        estimator: Fitted scikit-learn model.
        flat_vector: Preprocessed 1x784 feature vector (values in [0.0, 1.0]).

    Returns:
        {
            "predicted_digit": int,
            "confidence": float (e.g. 0.985),
            "probabilities": list of 10 floats (digits 0 through 9)
        }
    """
    if flat_vector.ndim == 1:
        flat_vector = flat_vector.reshape(1, -1)

    predicted_class = int(estimator.predict(flat_vector)[0])
    probabilities = [0.0] * 10

    if hasattr(estimator, "predict_proba"):
        raw_probs = estimator.predict_proba(flat_vector)[0]
        # Align with estimator.classes_ if present
        if hasattr(estimator, "classes_"):
            for cls_label, p in zip(estimator.classes_, raw_probs):
                cls_idx = int(cls_label)
                if 0 <= cls_idx < 10:
                    probabilities[cls_idx] = float(p)
        else:
            for idx in range(min(10, len(raw_probs))):
                probabilities[idx] = float(raw_probs[idx])
    elif hasattr(estimator, "decision_function"):
        raw_decision = estimator.decision_function(flat_vector)[0]
        prob_dist = softmax(raw_decision)
        if hasattr(estimator, "classes_"):
            for cls_label, p in zip(estimator.classes_, prob_dist):
                cls_idx = int(cls_label)
                if 0 <= cls_idx < 10:
                    probabilities[cls_idx] = float(p)
        else:
            for idx in range(min(10, len(prob_dist))):
                probabilities[idx] = float(prob_dist[idx])
    else:
        # Fallback if no probabilities or decision function
        probabilities[predicted_class] = 1.0

    # Ensure probabilities sum to 1.0
    total = sum(probabilities)
    if total > 0:
        probabilities = [p / total for p in probabilities]

    confidence = probabilities[predicted_class]

    return {
        "predicted_digit": predicted_class,
        "confidence": confidence,
        "probabilities": probabilities
    }


def predict_across_all_models(models_dict: Dict[str, Any], flat_vector: np.ndarray) -> Dict[str, Dict[str, Any]]:
    """
    Evaluates the input drawing across all loaded model estimators.
    """
    results = {}
    for name, est in models_dict.items():
        try:
            results[name] = predict_digit_probabilities(est, flat_vector)
        except Exception as e:
            results[name] = {
                "error": str(e),
                "predicted_digit": None,
                "confidence": 0.0,
                "probabilities": [0.0] * 10
            }
    return results
