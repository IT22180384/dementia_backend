"""
Test what the model predicts for 86% accuracy (43/50)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import joblib
import numpy as np

# Load models
model = joblib.load('src/models/game/risk_classifier/logistic_regression_model.pkl')
scaler = joblib.load('src/models/game/risk_classifier/feature_scaler.pkl')
encoder = joblib.load('src/models/game/risk_classifier/label_encoder.pkl')

print("\n" + "=" * 70)
print("TEST: 86% ACCURACY (43/50)")
print("=" * 70)

# Simulate 86% accuracy with RT around 220ms = 0.220 seconds
accuracy = 0.86
rt_seconds = 0.220

# Calculate SAC and IES
sac = accuracy / rt_seconds
ies = rt_seconds / accuracy

print(f"\nPerformance Metrics:")
print(f"   Accuracy: {accuracy:.2%}")
print(f"   RT: {rt_seconds:.3f}s ({rt_seconds*1000:.0f}ms)")
print(f"   SAC: {sac:.4f}")
print(f"   IES: {ies:.4f}")

# Create 14 features for the model
features = np.array([[
    sac,      # mean_sac
    0.0,      # slope_sac
    ies,      # mean_ies  
    0.0,      # slope_ies
    accuracy, # mean_acc
    rt_seconds, # mean_rt
    0.05,     # mean_var
    0.1,      # lstm_score
    sac,      # current_sac
    ies,      # current_ies
    0.0,      # slope_acc
    0.0,      # slope_rt
    0.01,     # std_sac
    0.02      # std_ies
]])

print(f"\n14 Model Features:")
for i, val in enumerate(features[0]):
    print(f"   Feature {i}: {val:.4f}")

# Scale and predict
X_scaled = scaler.transform(features)
probs = model.predict_proba(X_scaled)[0]
predicted_class = model.predict(X_scaled)[0]
predicted_label = encoder.inverse_transform([predicted_class])[0]

print(f"\n🎯 MODEL PREDICTION:")
print(f"   Predicted class: {predicted_class}")
print(f"   Predicted label: {predicted_label}")
print(f"   Probabilities:")
print(f"      HIGH:   {probs[0]:.4f} ({probs[0]*100:.1f}%)")
print(f"      LOW:    {probs[1]:.4f} ({probs[1]*100:.1f}%)")
print(f"      MEDIUM: {probs[2]:.4f} ({probs[2]*100:.1f}%)")

print(f"\n✅ EXPECTED: LOW or MEDIUM risk for 86% accuracy")
print(f"✅ ACTUAL: {predicted_label}")

if predicted_label == "HIGH":
    print("\n⚠️ WARNING: Model is predicting HIGH for good performance!")
    print("   This suggests the features being sent are incorrect.")
else:
    print(f"\n✅ Model is working correctly!")

print("=" * 70 + "\n")
