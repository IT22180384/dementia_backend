"""
Test script to verify risk classifier with low accuracy (4%) like in the screenshot
"""
import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("=" * 70)
print("Testing Risk Prediction with Low Accuracy (4%)")
print("=" * 70)

from src.models.game.model_registry import load_all_models, get_risk_classifier, get_risk_scaler, get_label_encoder

# Load models
print("\n1. Loading models...")
load_all_models()

risk_model = get_risk_classifier()
scaler = get_risk_scaler()
encoder = get_label_encoder()

print(f"   Model: {type(risk_model).__name__}")
print(f"   Scaler: {type(scaler).__name__ if scaler else 'None'}")
print(f"   Encoder: {type(encoder).__name__ if encoder else 'None'}")

# Simulate features from a game with 4% accuracy (2/50)
print("\n2. Simulating game with 4% accuracy (like screenshot)...")
accuracy = 0.04  # 4%
rt_adj_median = 2000  # Slow reaction time
sac = 0.0266  # From the logs
ies = 15.025  # High IES (poor performance)
variability = 0.8

# Build feature vector (14 features)
# Assuming this is the first session (no history)
features = np.array([[
    sac,          # mean_sac
    0.0,          # slope_sac  
    ies,          # mean_ies
    0.0,          # slope_ies
    accuracy,     # mean_accuracy
    rt_adj_median, # mean_rt
    variability,  # mean_variability
    0.0,          # lstm_score (dummy LSTM)
    sac,          # current_sac
    ies,          # current_ies
    0.0,          # slope_accuracy
    0.0,          # slope_rt
    0.0,          # std_sac
    0.0           # std_ies
]])

print(f"   Accuracy: {accuracy:.1%}")
print(f"   SAC: {sac:.4f}")
print(f"   IES: {ies:.4f}")
print(f"   RT (adjusted): {rt_adj_median:.0f}ms")

# Scale features
print("\n3. Scaling features...")
if scaler:
    features_scaled = scaler.transform(features)
    print("   ✓ Features scaled")
else:
    features_scaled = features
    print("   ⚠️ No scaler - using raw features")

# Make prediction
print("\n4. Making prediction...")
prediction = risk_model.predict(features_scaled)[0]
probabilities = risk_model.predict_proba(features_scaled)[0]

print(f"   Prediction class: {prediction}")
print(f"   Probabilities:")
print(f"     LOW: {probabilities[0]:.4f} ({probabilities[0]*100:.2f}%)")
print(f"     MEDIUM: {probabilities[1]:.4f} ({probabilities[1]*100:.2f}%)")
print(f"     HIGH: {probabilities[2]:.4f} ({probabilities[2]*100:.2f}%)")

if encoder:
    label = encoder.inverse_transform([prediction])[0]
    print(f"\n   ✓ Risk Level: {label}")
else:
    risk_labels = ["LOW", "MEDIUM", "HIGH"]
    label = risk_labels[prediction]
    print(f"\n   ✓ Risk Level: {label}")

print("\n" + "=" * 70)
print(f"RESULT: With 4% accuracy, model predicts: {label}")
print("=" * 70)

# Sanity check
if accuracy < 0.1 and label == "LOW":
    print("\n⚠️ WARNING: Model predicting LOW risk despite 4% accuracy!")
    print("This suggests the model might not be working correctly.")
elif accuracy < 0.1 and label == "HIGH":
    print("\n✅ EXPECTED: Low accuracy (4%) correctly classified as HIGH risk")
else:
    print(f"\n⚠️ Model predicted {label} for 4% accuracy - check if this is expected")
