"""
predict.py - Prediction utilities for Student Performance Predictor
"""

import os
import json
import numpy as np
import joblib

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# ── Load artifacts once (module-level cache) ───────────────────────────────────
_cache = {}

def _load_cache():
    if _cache:
        return
    with open(os.path.join(MODEL_DIR, 'metadata.json')) as f:
        _cache['metadata'] = json.load(f)
    _cache['scaler']   = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    _cache['encoders'] = joblib.load(os.path.join(MODEL_DIR, 'encoders.pkl'))
    _cache['best_reg'] = joblib.load(os.path.join(MODEL_DIR, 'best_regressor.pkl'))
    _cache['best_clf'] = joblib.load(os.path.join(MODEL_DIR, 'best_classifier.pkl'))

    # All individual models
    import glob
    _cache['reg_models'] = {}
    _cache['clf_models'] = {}
    for path in glob.glob(os.path.join(MODEL_DIR, 'reg_*.pkl')):
        name = os.path.basename(path).replace('reg_','').replace('.pkl','').replace('_',' ')
        if 'best' not in name:
            _cache['reg_models'][name] = joblib.load(path)
    for path in glob.glob(os.path.join(MODEL_DIR, 'clf_*.pkl')):
        name = os.path.basename(path).replace('clf_','').replace('.pkl','').replace('_',' ')
        if 'best' not in name:
            _cache['clf_models'][name] = joblib.load(path)


# ── Category maps (same as synthetic dataset values) ──────────────────────────────
CATEGORY_OPTIONS = {
    'gender': ['Female', 'Male'],
    'parental_education': ['None', 'High School', 'Intermediate', 'Bachelors', 'Masters', 'PhD'],
    'internet_access': ['Yes', 'No'],
    'extracurricular_activities': ['Yes', 'No'],
}

def preprocess_input(form_data: dict) -> np.ndarray:
    """
    Encode and scale a single student's inputs.
    """
    _load_cache()
    encoders = _cache['encoders']
    scaler   = _cache['scaler']
    meta     = _cache['metadata']
    feature_cols = meta['feature_cols']

    row = []
    for col in feature_cols:
        val = form_data.get(col)
        
        if col in meta['categorical_cols']:
            # Encode categorical
            encoded = encoders[col].transform([val])[0]
            row.append(encoded)
        else:
            # Numeric columns
            row.append(float(val))

    X = np.array(row).reshape(1, -1)
    X_scaled = scaler.transform(X)
    return X_scaled


def run_prediction(form_data: dict) -> dict:
    """
    Run predictions from both best models and all individual models.
    Returns a dict with grade prediction, pass/fail, and all model results.
    """
    _load_cache()
    X = preprocess_input(form_data)
    meta = _cache['metadata']

    # ── Best model predictions ─────────────────────────────────────────────
    grade_pred  = float(_cache['best_reg'].predict(X)[0])
    grade_pred  = max(0.0, min(100.0, grade_pred))
    pass_prob   = None
    if hasattr(_cache['best_clf'], 'predict_proba'):
        proba     = _cache['best_clf'].predict_proba(X)[0]
        pass_prob = float(proba[1])  # probability of passing
    pass_fail = int(_cache['best_clf'].predict(X)[0])

    # ── All regression models ─────────────────────────────────────────────
    reg_predictions = {}
    for name, model in _cache['reg_models'].items():
        pred = float(model.predict(X)[0])
        reg_predictions[name] = round(max(0, min(100, pred)), 2)

    # ── All classification models ─────────────────────────────────────────
    clf_predictions = {}
    for name, model in _cache['clf_models'].items():
        clf_predictions[name] = int(model.predict(X)[0])

    # ── Letter grade ──────────────────────────────────────────────────────
    def letter_grade(score):
        if score >= 90: return 'A+'
        if score >= 80: return 'A'
        if score >= 70: return 'B'
        if score >= 60: return 'C'
        if score >= 50: return 'D'
        return 'F'

    return {
        'grade_prediction':    round(grade_pred, 2),
        'letter_grade':        letter_grade(grade_pred),
        'pass_fail':           pass_fail,
        'pass_probability':    round(pass_prob * 100, 1) if pass_prob is not None else None,
        'best_regressor':      meta['best_regressor'],
        'best_classifier':     meta['best_classifier'],
        'all_reg_predictions': reg_predictions,
        'all_clf_predictions': clf_predictions,
    }


def get_model_metrics() -> dict:
    """Return stored evaluation metrics for all models."""
    _load_cache()
    meta = _cache['metadata']
    return {
        'regression':     meta['regression_results'],
        'classification': meta['classification_results'],
        'best_regressor': meta['best_regressor'],
        'best_classifier':meta['best_classifier'],
    }


def get_feature_info() -> dict:
    """Return feature categories and metadata."""
    _load_cache()
    return {
        'features':   _cache['metadata']['feature_cols'],
        'categories': CATEGORY_OPTIONS,
    }
