"""
SeedLab AI — Model Artifact Loader & Integrity Validator.
Safely loads pre-trained scikit-learn estimators and metadata without training inside Streamlit.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import io
import json
import zipfile
import joblib
import numpy as np


CANONICAL_FILENAMES = {
    "Random Forest": "random_forest.joblib",
    "MLP Neural Network": "mlp.joblib",
    "Logistic Regression": "logistic_regression.joblib"
}


def find_default_models_dir() -> Path:
    """Return the absolute path to the local project models directory."""
    current = Path(__file__).resolve().parent.parent / "models"
    return current


def extract_artifacts_from_zip(zip_bytes_or_buffer) -> Dict[str, bytes]:
    """Extract zip archive in-memory and return mapping of filename -> bytes."""
    extracted = {}
    with zipfile.ZipFile(zip_bytes_or_buffer, "r") as z:
        for file_info in z.infolist():
            # Skip directories or macOS __MACOSX noise
            if file_info.is_dir() or "__MACOSX" in file_info.filename or file_info.filename.startswith("."):
                continue
            name = Path(file_info.filename).name
            extracted[name] = z.read(file_info.filename)
    return extracted


def validate_and_load_estimator(
    model_bytes_or_path,
    expected_name: str
) -> Tuple[Optional[Any], Optional[str]]:
    """
    Safely load an estimator from file path or in-memory bytes.
    Validates scikit-learn estimator interface and input dimension (784).
    """
    try:
        if isinstance(model_bytes_or_path, (str, Path)):
            p = Path(model_bytes_or_path)
            if not p.exists():
                return None, f"Artifact file '{p.name}' not found."
            estimator = joblib.load(p)
        else:
            buffer = io.BytesIO(model_bytes_or_path)
            estimator = joblib.load(buffer)

        # Validate interface
        if not hasattr(estimator, "predict"):
            return None, f"Loaded object for '{expected_name}' does not implement the standard scikit-learn .predict() method."

        # Validate input features if available
        if hasattr(estimator, "n_features_in_") and estimator.n_features_in_ != 784:
            return None, f"Feature dimension mismatch: expected 784 features (28x28), but model expects {estimator.n_features_in_}."

        return estimator, None
    except Exception as e:
        return None, f"Error deserializing model: {str(e)}"


def load_model_suite(
    models_dir: Optional[Path] = None,
    uploaded_zip_file=None
) -> Dict[str, Any]:
    """
    Loads all available models and metadata from local directory or an uploaded ZIP.

    Returns dict:
        {
            "metadata": dict or None,
            "models": { "Random Forest": estimator, ... },
            "errors": { "MLP Neural Network": "error msg", ... },
            "source": "local_directory" | "uploaded_zip" | "none",
            "is_ready": bool
        }
    """
    result = {
        "metadata": None,
        "models": {},
        "errors": {},
        "source": "none",
        "is_ready": False
    }

    # Case 1: Load from uploaded ZIP file if provided
    if uploaded_zip_file is not None:
        try:
            zip_bytes = uploaded_zip_file.read()
            files_dict = extract_artifacts_from_zip(io.BytesIO(zip_bytes))
            result["source"] = f"Uploaded ZIP ({uploaded_zip_file.name})"

            # Parse metadata
            if "metadata.json" in files_dict:
                try:
                    result["metadata"] = json.loads(files_dict["metadata.json"].decode("utf-8"))
                except Exception as je:
                    result["errors"]["metadata"] = f"Failed to parse metadata.json: {je}"
            else:
                result["errors"]["metadata"] = "ZIP archive is missing metadata.json."

            # Load each canonical model
            for model_name, filename in CANONICAL_FILENAMES.items():
                if filename in files_dict:
                    est, err = validate_and_load_estimator(files_dict[filename], model_name)
                    if est is not None:
                        result["models"][model_name] = est
                    else:
                        result["errors"][model_name] = err
                else:
                    result["errors"][model_name] = f"Archive missing '{filename}'"

            result["is_ready"] = len(result["models"]) > 0
            return result
        except Exception as ze:
            result["errors"]["zip"] = f"Failed to read ZIP archive: {ze}"
            return result

    # Case 2: Load from local models/ directory
    if models_dir is None:
        models_dir = find_default_models_dir()

    if models_dir.exists() and models_dir.is_dir():
        result["source"] = f"Local Directory ({models_dir.name}/)"
        meta_path = models_dir / "metadata.json"
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    result["metadata"] = json.load(f)
            except Exception as je:
                result["errors"]["metadata"] = f"Failed to load metadata.json: {je}"
        else:
            result["errors"]["metadata"] = "Local directory missing metadata.json."

        for model_name, filename in CANONICAL_FILENAMES.items():
            model_path = models_dir / filename
            if model_path.exists():
                est, err = validate_and_load_estimator(model_path, model_name)
                if est is not None:
                    result["models"][model_name] = est
                else:
                    result["errors"][model_name] = err
            else:
                result["errors"][model_name] = f"Missing '{filename}' in {models_dir.name}/"

        result["is_ready"] = len(result["models"]) > 0

    return result
