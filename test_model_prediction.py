"""
Test to check what the model actually predicts for good performance
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
print("MODEL PREDICTION TEST")
print("=" * 70)

print(f"\nModel type: {type(model).__name__}")
print(f"Model classes (numeric): {model.classes_}")
print(f"Encoder classes (labels): {encoder.classes_}")

# Create sample data representing GOOD performance (66% accuracy)
# 14 features: mean_sac, slope_sac, mean_ies, slope_ies, mean_acc, mean_rt, mean_var, lstm_score,
#              current_sac, current_ies, slope_acc, slope_rt, std_sac, std_ies

good_performance = np.array([[
    0.02,    # mean_sac (low = good)
    -0.01,   # slope_sac (decreasing = improving)
    120.0,   # mean_ies (lower = good)
    -5.0,    # slope_ies (decreasing = improving)
    0.66,    # mean_acc (66% accuracy - GOOD)
    600.0,   # mean_rt (reasonable reaction time)
    50.0,    # mean_var (low variability)
    0.15,    # lstm_score (low decline)
    0.015,   # current_sac
    115.0,   # current_ies
    0.02,    # slope_acc (improving)
    -10.0,   # slope_rt (getting faster)
    0.01,    # std_sac
    15.0     # std_ies
]])

print(f"\n📊 Test Input (Good Performance - 66% accuracy):")
print(f"   Accuracy: 0.66 (66%)")
print(f"   SAC: 0.015 (low errors)")
print(f"   IES: 115 (good speed-accuracy)")
print(f"   RT: 600ms")

# Scale features
X_scaled = scaler.transform(good_performance)
print(f"\n⚙️ Features scaled")

# Get predictions
probs = model.predict_proba(X_scaled)[0]
predicted_class = model.predict(X_scaled)[0]
predicted_label = encoder.inverse_transform([predicted_class])[0]

print(f"\n🎯 RAW MODEL OUTPUT:")
print(f"   Predicted class (numeric): {predicted_class}")
print(f"   Probabilities for classes {model.classes_}: {probs}")
print(f"   Predicted label: {predicted_label}")

print(f"\n📋 PROBABILITY BREAKDOWN:")
print(f"   P(class 0 = {encoder.classes_[0]}): {probs[0]:.4f} ({probs[0]*100:.1f}%)")
print(f"   P(class 1 = {encoder.classes_[1]}): {probs[1]:.4f} ({probs[1]*100:.1f}%)")
print(f"   P(class 2 = {encoder.classes_[2]}): {probs[2]:.4f} ({probs[2]*100:.1f}%)")

print(f"\n✅ FINAL PREDICTION: {predicted_label}")

# Now test with POOR performance
print("\n" + "=" * 70)
poor_performance = np.array([[
    0.08,    # mean_sac (high = bad)
    0.02,    # slope_sac (increasing = worsening)
    200.0,   # mean_ies (higher = bad)
    10.0,    # slope_ies (increasing = worsening)
    0.40,    # mean_acc (40% accuracy - POOR)
    900.0,   # mean_rt (slow reaction time)
    100.0,   # mean_var (high variability)
    0.35,    # lstm_score (high decline)
    0.09,    # current_sac
    220.0,   # current_ies
    -0.05,   # slope_acc (worsening)
    20.0,    # slope_rt (getting slower)
    0.03,    # std_sac
    30.0     # std_ies
]])

print(f"📊 Test Input (Poor Performance - 40% accuracy):")
print(f"   Accuracy: 0.40 (40%)")
print(f"   SAC: 0.09 (high errors)")
print(f"   IES: 220 (poor speed-accuracy)")
print(f"   RT: 900ms")

X_scaled_poor = scaler.transform(poor_performance)
probs_poor = model.predict_proba(X_scaled_poor)[0]
predicted_class_poor = model.predict(X_scaled_poor)[0]
predicted_label_poor = encoder.inverse_transform([predicted_class_poor])[0]

print(f"\n🎯 RAW MODEL OUTPUT:")
print(f"   Predicted class (numeric): {predicted_class_poor}")
print(f"   Predicted label: {predicted_label_poor}")
print(f"   Probabilities: HIGH={probs_poor[0]:.3f}, LOW={probs_poor[1]:.3f}, MED={probs_poor[2]:.3f}")

print(f"\n✅ FINAL PREDICTION: {predicted_label_poor}")
print("=" * 70)
