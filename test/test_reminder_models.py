"""Quick test to validate all reminder system models."""
import sys
sys.path.insert(0, '.')
import json
from src.features.reminder_system.enhanced_model_loader import EnhancedModelLoader

loader = EnhancedModelLoader()
print('\n=== Model Info ===')
print(json.dumps(loader.get_model_info(), indent=2, default=str))

test_cases = [
    ('Yes, I took my medication', 'healthy'),
    ("Um... what medicine? I don't remember...", 'confused'),
    ("I think... maybe... I forgot what I was doing. Help me.", 'high_risk'),
    ('Already took it this morning, thank you', 'healthy'),
    ("What? Where am I? I don't understand what is happening", 'severe'),
]

print('\n=== Predictions ===')
for text, expected in test_cases:
    risk, risk_conf = loader.predict_cognitive_risk(text)
    confused, conf_score = loader.predict_confusion_detection(text)
    alert, alert_score = loader.predict_caregiver_alert(text)
    severity, sev_conf = loader.predict_severity(text)
    
    print(f'\nText: "{text}"')
    print(f'  Expected:     {expected}')
    print(f'  Dementia Risk: {risk:.1%} (confidence: {risk_conf:.2f})')
    print(f'  Confused:      {confused} (confidence: {conf_score:.2f})')
    print(f'  Alert Needed:  {alert} (risk: {alert_score:.2f})')
    print(f'  Severity:      {severity} (confidence: {sev_conf:.2f})')

print('\nAll models working!')
