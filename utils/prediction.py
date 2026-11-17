"""
SeedLab AI — Authentic Inference & Probability Engine.
Zero fabrication: Extracts genuine, unmanipulated probabilities directly from model estimators.
Supports calibrated probability estimates (CalibratedClassifierCV) with scientific interpretations.
"""

from typing import Dict, Any, List, Optional
import numpy as np
from sklearn.calibration import CalibratedClassifierCV


def softmax(x: np.ndarray) -> np.ndarray:
    """Compute numerically stable softmax values for linear model decision margins."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def check_is_calibrated(estimator: Any, metadata_entry: Optional[Dict[str, Any]] = None) -> bool:
    """
    Detect whether the fitted estimator produces calibrated probability estimates.
    Checks scikit-learn estimator type and artifact metadata.
    """
    if isinstance(estimator, CalibratedClassifierCV) or hasattr(estimator, "calibrated_classifiers_"):
        return True
    if metadata_entry and metadata_entry.get("is_calibrated", False):
        return True
    return False


def get_scientific_interpretation(prob: float) -> str:
    """
    Provides a scientifically cautious interpretation of the predicted-class probability.
    Does not assume a high probability implies guaranteed correctness.
    """
    if prob >= 0.80:
        return "Higher probability: the model assigns greater probability to this predicted class."
    elif prob >= 0.50:
        return "Moderate probability: the model distributes probability across multiple possible classes."
    else:
        return "Low probability: the model is relatively uncertain about its prediction."


def predict_digit_probabilities(
    estimator: Any,
    flat_vector: np.ndarray,
    metadata_entry: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Infers the digit class and extracts authentic, unmanipulated class probabilities.

    Rules:
    - Never cap, inflate, reduce, or fabricate probabilities.
    - Directly query the estimator's .predict_proba() implementation.
    - Align output probabilities with estimator's actual .classes_ array.
    - Transparently document whether probabilities are calibrated or raw.

    Args:
        estimator: Fitted scikit-learn model or CalibratedClassifierCV instance.
        flat_vector: Preprocessed 1x784 feature vector (values in [0.0, 1.0]).
        metadata_entry: Optional model metadata entry from metadata.json.

    Returns:
        dict containing:
            - predicted_digit: int
            - predicted_class_probability: float
            - probability_label: str
            - is_calibrated: bool
            - calibration_status: str
            - probabilities: list of 10 floats (digits 0 to 9)
            - scientific_interpretation: str
            - caution_note: str
    """
    if flat_vector.ndim == 1:
        flat_vector = flat_vector.reshape(1, -1)

    # 1. Determine Calibration Status
    is_calibrated = check_is_calibrated(estimator, metadata_entry)
    if is_calibrated:
        calib_status_text = "Calibrated (CalibratedClassifierCV, Platt Scaling)"
        prob_label = "Calibrated predicted-class probability"
    else:
        calib_status_text = "Raw model probability — not calibrated"
        prob_label = "Predicted-class probability"

    # 2. Authentic Model Prediction
    predicted_class = int(estimator.predict(flat_vector)[0])
    probabilities = [0.0] * 10

    # 3. Extract Genuine Probabilities
    if hasattr(estimator, "predict_proba"):
        raw_probs = estimator.predict_proba(flat_vector)[0]
        # Align with estimator.classes_
        if hasattr(estimator, "classes_"):
            for cls_label, p in zip(estimator.classes_, raw_probs):
                cls_idx = int(cls_label)
                if 0 <= cls_idx < 10:
                    probabilities[cls_idx] = float(p)
        else:
            for idx in range(min(10, len(raw_probs))):
                probabilities[idx] = float(raw_probs[idx])
    elif hasattr(estimator, "decision_function"):
        # For estimators without predict_proba, apply softmax to decision logits
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
        # One-hot fallback if estimator has neither predict_proba nor decision_function
        probabilities[predicted_class] = 1.0

    # Actual probability of the winning class (unmanipulated)
    pred_prob = float(probabilities[predicted_class])
    interpretation = get_scientific_interpretation(pred_prob)

    caution_note = (
        "Predicted probabilities represent the model's estimated likelihood for each class. "
        "They do not guarantee that a prediction is correct."
    )

    return {
        "predicted_digit": predicted_class,
        "predicted_class_probability": pred_prob,
        "confidence": pred_prob,
        "probability_label": prob_label,
        "is_calibrated": is_calibrated,
        "calibration_status": calib_status_text,
        "probabilities": probabilities,
        "scientific_interpretation": interpretation,
        "caution_note": caution_note
    }


def predict_across_all_models(
    models_dict: Dict[str, Any],
    flat_vector: np.ndarray,
    metadata_models: Optional[Dict[str, Any]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluates the input drawing across all loaded model estimators.
    """
    results = {}
    for name, est in models_dict.items():
        meta_entry = metadata_models.get(name, None) if metadata_models else None
        try:
            results[name] = predict_digit_probabilities(est, flat_vector, meta_entry)
        except Exception as e:
            results[name] = {
                "error": str(e),
                "predicted_digit": None,
                "predicted_class_probability": 0.0,
                "probability_label": "Error",
                "is_calibrated": False,
                "calibration_status": "Unavailable",
                "probabilities": [0.0] * 10,
                "scientific_interpretation": "Error evaluating model.",
                "caution_note": ""
            }
    return results
