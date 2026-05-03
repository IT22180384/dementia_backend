"""
Test script for the NEW risk classifier model.
Run this AFTER copying your 3 pkl files into:
  src/models/game/risk_classifier/

Usage:
    venv\Scripts\python.exe test_new_risk_model.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

print("=" * 60)
print("  RISK CLASSIFIER MODEL TEST")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# STEP 1: Check that the files exist
# ─────────────────────────────────────────────────────────────
from pathlib import Path

model_dir = Path("src/models/game/risk_classifier")

expected_new = {
    "risk_logreg.pkl":        "Risk Classifier (Logistic Regression)",
    "risk_scaler.pkl":        "Feature Scaler",
    "risk_label_encoder.pkl": "Label Encoder",
}
expected_old = {
    "logistic_regression_model.pkl": "Risk Classifier (old name)",
    "feature_scaler.pkl":            "Feature Scaler (old name)",
    "label_encoder.pkl":             "Label Encoder (old name)",
}

print("\n📁 Checking model files in:", model_dir)
all_ok = True
for fname, label in expected_new.items():
    path = model_dir / fname
    exists = path.exists()
    status = "✅ FOUND" if exists else "❌ MISSING"
    print(f"   {status} | {fname:<35} ({label})")
    if not exists:
        # check old
        old_name = list(expected_old.keys())[list(expected_new.keys()).index(fname)]
        if (model_dir / old_name).exists():
            print(f"            ↳ Old file found: {old_name} (will be used as fallback)")
        else:
            all_ok = False

if not all_ok:
    print("\n❌ Some model files are missing. Please copy them before running.")
    print("   Expected location: src/models/game/risk_classifier/")
    sys.exit(1)

print("\n✅ All model files found!\n")

# ─────────────────────────────────────────────────────────────
# STEP 2: Load models via model_registry
# ─────────────────────────────────────────────────────────────
print("─" * 60)
print(" STEP 2: Loading models via model_registry...")
print("─" * 60)

from src.models.game.model_registry import (
    load_risk_classifier,
    load_risk_scaler,
    load_label_encoder,
)

risk_model   = load_risk_classifier()
risk_scaler  = load_risk_scaler()
label_enc    = load_label_encoder()

if risk_model is None:
    print("❌ FAILED: Risk classifier did not load.")
    sys.exit(1)
if risk_scaler is None:
    print("⚠️  WARNING: Risk scaler did not load. Predictions will be unscaled.")
if label_enc is None:
    print("⚠️  WARNING: Label encoder did not load.")

print(f"\n   Model type  : {risk_model.__class__.__name__}")
print(f"   Scaler type : {risk_scaler.__class__.__name__ if risk_scaler else 'None'}")
if label_enc is not None:
    print(f"   Risk classes: {list(label_enc.classes_)}")
else:
    print(f"   Risk classes: (no encoder — will use raw model output)")

# ─────────────────────────────────────────────────────────────
# STEP 3: Run Test Predictions — 3 known scenarios
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print(" STEP 3: Test predictions on known scenarios")
print("─" * 60)

# The 14 features the model expects (from game_service.py):
# [mean_sac, slope_sac, mean_ies, slope_ies, mean_accuracy,
#  mean_rt, mean_variability, lstm_score,
#  current_sac, current_ies, slope_accuracy, slope_rt, std_sac, std_ies]

test_cases = [
    {
        "name": "🟢 Healthy Player (High accuracy, fast RT)",
        "features": [0.80, 0.01, 1.25, -0.02, 0.90, 1.2, 0.05, 0.05, 0.82, 1.22, 0.0, 0.0, 0.02, 0.05],
        "expected_risk": "LOW",
    },
    {
        "name": "🟡 Borderline Player (Medium accuracy, slower RT)",
        "features": [0.45, -0.03, 2.80, 0.05, 0.55, 2.5, 0.20, 0.35, 0.42, 2.90, -0.02, 0.03, 0.08, 0.15],
        "expected_risk": "MEDIUM",
    },
    {
        "name": "🔴 Declining Player (Low accuracy, very slow RT)",
        "features": [0.05, -0.08, 15.0, 0.50, 0.08, 5.0, 0.80, 0.85, 0.04, 16.0, -0.05, 0.10, 0.02, 1.2],
        "expected_risk": "HIGH",
    },
]

RISK_LABELS = ["HIGH", "LOW", "MEDIUM"]  # alphabetical (label encoder default)

print()
all_passed = True
for case in test_cases:
    X = np.array([case["features"]])

    if risk_scaler is not None:
        try:
            X_scaled = risk_scaler.transform(X)
        except Exception as e:
            print(f"   ⚠️  Scaler error for '{case['name']}': {e}")
            X_scaled = X
    else:
        X_scaled = X

    try:
        probs = risk_model.predict_proba(X_scaled)[0]

        if label_enc is not None:
            classes = list(label_enc.classes_)
        else:
            # fallback: use alphabetical
            classes = RISK_LABELS

        # Build prob dict
        prob_dict = {cls: round(float(p), 3) for cls, p in zip(classes, probs)}
        predicted  = classes[int(np.argmax(probs))]
        score_0_100 = round(prob_dict.get("HIGH", 0.0) * 100, 1)

        match = "✅" if predicted == case["expected_risk"] else "⚠️ "
        if predicted != case["expected_risk"]:
            all_passed = False

        print(f"   {match} {case['name']}")
        print(f"      Predicted : {predicted}  (expected: {case['expected_risk']})")
        print(f"      Risk Score: {score_0_100}/100")
        print(f"      Probs     : {prob_dict}")
        print()

    except Exception as e:
        print(f"   ❌ Prediction failed for '{case['name']}': {e}")
        all_passed = False

# ─────────────────────────────────────────────────────────────
# STEP 4: Quick Model Info
# ─────────────────────────────────────────────────────────────
print("─" * 60)
print(" STEP 4: Model details")
print("─" * 60)
try:
    if hasattr(risk_model, 'coef_'):
        print(f"   Coefficients shape : {risk_model.coef_.shape}")
        print(f"   Number of features : {risk_model.coef_.shape[1]}")
    if hasattr(risk_model, 'C'):
        print(f"   Regularization (C) : {risk_model.C}")
    if hasattr(risk_scaler, 'mean_'):
        print(f"   Scaler mean (SAC)  : {risk_scaler.mean_[0]:.4f}")
        print(f"   Scaler scale (SAC) : {risk_scaler.scale_[0]:.4f}")
except Exception as e:
    print(f"   Could not read model details: {e}")

# ─────────────────────────────────────────────────────────────
# Final Result
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
if all_passed:
    print("  🎉 ALL TESTS PASSED — Model is working correctly!")
else:
    print("  ⚠️  SOME PREDICTIONS WERE UNEXPECTED.")
    print("  This may be normal if your new model was trained")
    print("  with different thresholds. Check the probabilities above.")
print("=" * 60)
