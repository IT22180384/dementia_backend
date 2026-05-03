"""
Test with EXACT features from the screenshot
"""
import sys
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("=" * 70)
print("Testing with EXACT features from game screenshot")
print("=" * 70)

from src.models.game.model_registry import load_all_models, get_risk_classifier, get_risk_scaler

# Load models
print("\n1. Loading models...")
load_all_models()

risk_model = get_risk_classifier()
scaler = get_risk_scaler()

# EXACT features from screenshot
print("\n2. Using EXACT features from screenshot:")
print("   Score: 3/50 = 6% accuracy")
print("   SAC: 0.039")
print("   IES: 15.025") 
print("   RT: 1502ms")

# Build feature vector (14 features) - first session so no history
features = np.array([[
    0.039,        # mean_sac (same as current since first session)
    0.0,          # slope_sac (no history)
    15.025,       # mean_ies
    0.0,          # slope_ies
    0.06,         # mean_accuracy (6%)
    1.502,        # mean_rt (1502ms = 1.502s)
    0.164,        # variability (164ms from screenshot)
    0.0,          # lstm_score (dummy)
    0.039,        # current_sac
    15.025,       # current_ies
    0.0,          # slope_accuracy
    0.0,          # slope_rt
    0.0,          # std_sac
    0.0           # std_ies
]])

print(f"\n3. Feature vector shape: {features.shape}")
print(f"   Features: {features[0]}")

# Scale
if scaler:
    features_scaled = scaler.transform(features)
    print(f"\n4. After scaling:")
    print(f"   Scaled[0-3]: {features_scaled[0, :4]}")
else:
    features_scaled = features

# Predict
prediction = risk_model.predict(features_scaled)[0]
probabilities = risk_model.predict_proba(features_scaled)[0]

print(f"\n5. MODEL PREDICTION:")
print(f"   Class: {prediction}")
print(f"   Probabilities [HIGH, LOW, MED]: {probabilities}")
print(f"   HIGH: {probabilities[0]:.4f} ({probabilities[0]*100:.2f}%)")
print(f"   LOW: {probabilities[1]:.4f} ({probabilities[1]*100:.2f}%)")
print(f"   MEDIUM: {probabilities[2]:.4f} ({probabilities[2]*100:.2f}%)")

risk_labels = ["HIGH", "LOW", "MEDIUM"]
predicted_label = risk_labels[prediction]

print(f"\n6. RESULT: {predicted_label}")
print(f"   Score: {probabilities[0]*100:.0f}/100")

print("\n" + "=" * 70)
if predicted_label == "LOW":
    print("⚠️ MODEL PREDICTS LOW RISK FOR 6% ACCURACY!")
    print("This means your model was trained with data where:")
    print("  - Low SAC (0.039) is associated with LOW risk")
    print("  - The model might be looking at SAC as 'low error cost'")
    print("  - rather than 'poor performance'")
    print("\nThe model learned the wrong pattern from training data!")
else:
    print(f"✅ Model correctly predicts {predicted_label} risk")
print("=" * 70)
