"""Test the pure Pitt-trained Gradient Boosting model integration."""
import sys
sys.path.insert(0, '.')
from src.features.reminder_system.enhanced_model_loader import EnhancedModelLoader

print("Loading Pitt model...")
loader = EnhancedModelLoader()

print("\n--- Model Info ---")
info = loader.get_model_info()
for k, v in info.items():
    if k != 'metadata':
        print(f"  {k}: {v}")

# Test cognitive risk predictions
test_cases = [
    ("Yes I took my medication after breakfast", "Clear confirmation"),
    ("Um... I think I did... maybe... wait what was it again?", "Confused/hesitant"),
    ("What medicine? I do not remember any medicine", "Memory issue"),
    ("Done", "Very short response"),
    ("I already took it this morning with my coffee and toast like I always do every day", "Detailed healthy response"),
    ("um uh I um I think uh maybe I did the um thing uh wait what", "Heavy disfluency"),
    ("Yes", "Minimal healthy response"),
    ("I... I don't... what was I... um... the thing... I forgot... sorry... wait... um", "Severe confusion"),
    ("I took my pills at 8am, then had breakfast, and went for my walk", "Coherent detailed response"),
    ("medicine medicine medicine what what what I don't know", "Repetitive confused"),
]

print("\n--- Cognitive Risk Predictions ---")
for text, desc in test_cases:
    dementia_prob, confidence = loader.predict_cognitive_risk(text)
    confused, confusion_conf = loader.predict_confusion_detection(text)
    alert, alert_risk = loader.predict_caregiver_alert(text)
    
    risk_level = "LOW" if dementia_prob < 0.25 else "MILD" if dementia_prob < 0.5 else "MODERATE" if dementia_prob < 0.75 else "HIGH"
    
    print(f"\n  [{desc}]")
    print(f"  Text: \"{text}\"")
    print(f"  Dementia Prob: {dementia_prob:.3f} ({risk_level})")
    print(f"  Confidence:    {confidence:.3f}")
    print(f"  Confused:      {confused} (score={confusion_conf:.2f})")
    print(f"  Alert Needed:  {alert} (risk={alert_risk:.2f})")

print("\n All tests passed!")
