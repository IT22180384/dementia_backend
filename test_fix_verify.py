"""Quick verification that the fixed risk model predicts correctly."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.game.model_registry import load_all_models
from src.services.game_service import predict_risk

load_all_models()

cases = [
    ("User session (50% acc, 1406ms RT, var=607ms)", {"accuracy":0.50,"errorRate":0.50,"rtAdjMedian":1.406,"sac":0.3556,"ies":2.812,"variability":0.6072}, 0.0, "MEDIUM"),
    ("Healthy (86% acc, 220ms RT)", {"accuracy":0.86,"errorRate":0.14,"rtAdjMedian":0.22,"sac":3.9,"ies":0.26,"variability":0.05}, 0.0, "LOW"),
    ("Extreme decline (20% acc, 5s RT, var=900ms)", {"accuracy":0.20,"errorRate":0.80,"rtAdjMedian":5.00,"sac":0.04,"ies":25.0,"variability":0.90}, 0.85, "HIGH"),
    ("Borderline (55% acc, 2.5s RT)", {"accuracy":0.55,"errorRate":0.45,"rtAdjMedian":2.5,"sac":0.22,"ies":4.55,"variability":0.35}, 0.30, "MEDIUM"),
]

all_ok = True
for name, feat, lstm, expected in cases:
    result = predict_risk([], feat, lstm)
    predicted = result["riskLevel"]
    score = result["riskScore0_100"]
    ok = "OK" if predicted == expected else "FAIL"
    if predicted != expected: all_ok = False
    print(f"[{ok}] {name}")
    print(f"     Predicted: {predicted} (expected {expected}) | score={score}/100 | probs={result['riskProbability']}")

print()
print("ALL TESTS PASSED" if all_ok else "SOME TESTS FAILED")
