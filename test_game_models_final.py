"""
Final Integration Test — Game ML Models
========================================
Tests:
  1. Model files exist on disk
  2. Models load via model_registry (correct filenames, correct classes)
  3. Risk classifier feature count matches game_service (14 features)
  4. Risk predictions are sensible for known scenarios
  5. LSTM path degrades gracefully when TF is not installed
  6. Full predict_risk() pipeline end-to-end

Run from project root:
    .\\venv\\Scripts\\python.exe test_game_models_final.py
"""
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"
results = {}


def section(title):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


# ============================================================
# TEST 1 — Files exist
# ============================================================
section("TEST 1: Model Files Exist on Disk")
from pathlib import Path

required_files = {
    "risk_logreg.pkl":        Path("src/models/game/risk_classifier/risk_logreg.pkl"),
    "risk_scaler.pkl":        Path("src/models/game/risk_classifier/risk_scaler.pkl"),
    "risk_label_encoder.pkl": Path("src/models/game/risk_classifier/risk_label_encoder.pkl"),
    "lstm_model.keras":       Path("src/models/game/lstm_model/lstm_model.keras"),
    "lstm_scaler.pkl":        Path("src/models/game/lstm_model/lstm_scaler.pkl"),
}

all_exist = True
for name, path in required_files.items():
    exists = path.exists()
    status = "✅" if exists else "❌"
    print(f"  {status} {name}: {path}")
    if not exists:
        all_exist = False

results["files_exist"] = all_exist
print(f"\n  Result: {'PASS' if all_exist else 'FAIL'}")


# ============================================================
# TEST 2 — Model Registry loads correctly
# ============================================================
section("TEST 2: Model Registry — Load All Models")
try:
    from src.models.game.model_registry import (
        load_all_models, get_risk_classifier, get_risk_scaler,
        get_label_encoder, get_lstm_model, get_lstm_scaler
    )
    load_all_models()

    clf   = get_risk_classifier()
    scl   = get_risk_scaler()
    enc   = get_label_encoder()
    lstm  = get_lstm_model()
    lscl  = get_lstm_scaler()

    print(f"  Risk classifier : {clf.__class__.__name__   if clf else 'None'}")
    print(f"  Risk scaler     : {scl.__class__.__name__   if scl else 'None'}")
    print(f"  Label encoder   : {list(enc.classes_)        if enc else 'None'}")
    print(f"  LSTM model      : {lstm.__class__.__name__   if lstm else 'None (TF not installed – expected)'}")
    print(f"  LSTM scaler     : {lscl.__class__.__name__   if lscl else 'None'}")

    ok = clf is not None and scl is not None and enc is not None
    results["registry_load"] = ok
    print(f"\n  Result: {'PASS' if ok else 'FAIL'}")
except Exception as e:
    print(f"  ❌ Exception: {e}")
    import traceback; traceback.print_exc()
    results["registry_load"] = False


# ============================================================
# TEST 3 — Feature count matches
# ============================================================
section("TEST 3: Risk Classifier Feature Count == 14")
try:
    clf = get_risk_classifier()
    scl = get_risk_scaler()
    n   = clf.n_features_in_ if clf else -1
    n_s = scl.n_features_in_ if scl else -1

    print(f"  Classifier n_features_in_: {n}  (expected 14)")
    print(f"  Scaler n_features_in_    : {n_s}  (expected 14)")

    ok = (n == 14) and (n_s == 14)
    results["feature_count"] = ok
    print(f"\n  Result: {'PASS' if ok else 'FAIL'}")
except Exception as e:
    print(f"  ❌ Exception: {e}")
    results["feature_count"] = False


# ============================================================
# TEST 4 — Risk predictions on known scenarios
# ============================================================
section("TEST 4: Risk Predictions on Known Scenarios")

# 14-feature vector builder (mirroring game_service.extract_risk_features first-session)
def make_features(accuracy, rt_s, variability, lstm_score=0.0):
    sac = accuracy / max(rt_s, 0.001)
    ies = rt_s / max(accuracy, 0.001)
    return np.array([[
        sac,         # mean_sac
        0.0,         # slope_sac
        ies,         # mean_ies
        0.0,         # slope_ies
        accuracy,    # mean_accuracy
        rt_s,        # mean_rt
        variability, # mean_variability
        lstm_score,  # lstm_decline_score
        sac,         # current_sac
        ies,         # current_ies
        0.0,         # slope_accuracy
        0.0,         # slope_rt
        0.0,         # std_sac
        0.0,         # std_ies
    ]])

RISK_LABELS = ["HIGH", "LOW", "MEDIUM"]

def predict(X):
    clf = get_risk_classifier()
    scl = get_risk_scaler()
    if scl:
        X = scl.transform(X)
    probs = clf.predict_proba(X)[0]
    return RISK_LABELS[int(np.argmax(probs))], dict(zip(RISK_LABELS, np.round(probs, 3)))

test_cases = [
    ("Healthy  (90% acc, 0.5s RT,  low var)",    0.90, 0.50, 0.05,  "LOW"),
    ("Moderate (55% acc, 1.5s RT,  med var)",    0.55, 1.50, 0.30,  "MEDIUM"),
    ("Decline  (20% acc, 4.0s RT,  high var)",   0.20, 4.00, 0.80,  "HIGH"),
]

all_passed = True
try:
    for name, acc, rt, var, expected in test_cases:
        X  = make_features(acc, rt, var)
        pred, probs = predict(X)

        # Apply same rule-based floor as game_service.py
        if acc < 0.35 or (rt / max(acc, 0.001)) > 8.0 or (acc < 0.40 and var > 0.60):
            if pred in ("LOW", "MEDIUM"):
                pred = "HIGH"  # rule override

        ok = "✅" if pred == expected else "⚠️ "
        if pred != expected:
            all_passed = False
        print(f"  {ok} {name}")
        print(f"      Predicted: {pred}  (expected {expected})  probs={probs}")

    results["predictions"] = all_passed
    print(f"\n  Result: {'PASS' if all_passed else 'WARN — check probs above'}")
except Exception as e:
    print(f"  ❌ Exception: {e}")
    import traceback; traceback.print_exc()
    results["predictions"] = False


# ============================================================
# TEST 5 — LSTM degrades gracefully without TF
# ============================================================
section("TEST 5: LSTM Graceful Degradation (No TensorFlow)")
try:
    from src.services.game_service import predict_lstm_decline

    fake_sessions = [
        {"features": {"sac": 0.5, "ies": 2.0, "accuracy": 0.70,
                      "rtAdjMedian": 0.7, "variability": 0.15}} for _ in range(5)
    ]

    score = predict_lstm_decline(fake_sessions)
    print(f"  predict_lstm_decline(5 sessions) = {score}")
    print(f"  Type: {type(score).__name__}  — expected float, no exception")

    ok = isinstance(score, float)
    results["lstm_graceful"] = ok
    print(f"\n  Result: {'PASS' if ok else 'FAIL'}")
except Exception as e:
    print(f"  ❌ Exception: {e}")
    import traceback; traceback.print_exc()
    results["lstm_graceful"] = False


# ============================================================
# TEST 6 — Full predict_risk() pipeline
# ============================================================
section("TEST 6: Full predict_risk() Pipeline (from game_service)")
try:
    from src.services.game_service import predict_risk

    high_risk_features = {
        "accuracy":    0.06,
        "sac":         0.039,
        "ies":         15.025,
        "rtAdjMedian": 1.502,
        "variability": 0.164,
    }

    ok_features = {
        "accuracy":    0.88,
        "sac":         0.80,
        "ies":         1.25,
        "rtAdjMedian": 0.55,
        "variability": 0.08,
    }

    for label, features, expected_risk in [
        ("6% accuracy (HIGH expected) ", high_risk_features, "HIGH"),
        ("88% accuracy (LOW expected) ", ok_features,        "LOW"),
    ]:
        result = predict_risk([], features, 0.0)

        risk_level  = result.get("riskLevel", "?")
        risk_score  = result.get("riskScore0_100", result.get("riskScore", "?"))
        probs       = result.get("riskProbabilities", result.get("probabilities", {}))
        model_used  = result.get("modelUsed", "unknown")

        ok = "✅" if risk_level == expected_risk else "⚠️ "
        print(f"\n  {ok} {label}")
        print(f"      riskLevel : {risk_level}  (expected {expected_risk})")
        print(f"      riskScore : {risk_score}")
        print(f"      probs     : {probs}")
        print(f"      modelUsed : {model_used}")

        if risk_level != expected_risk:
            results["pipeline"] = False

    if "pipeline" not in results:
        results["pipeline"] = True

    print(f"\n  Result: {'PASS' if results['pipeline'] else 'WARN — see above'}")

except RuntimeError as re:
    # Expected if model truly fails — but should work now
    print(f"  ❌ RuntimeError: {re}")
    results["pipeline"] = False
except Exception as e:
    print(f"  ❌ Exception: {e}")
    import traceback; traceback.print_exc()
    results["pipeline"] = False


# ============================================================
# SUMMARY
# ============================================================
section("SUMMARY")
pass_count = sum(1 for v in results.values() if v)
total = len(results)

for test_name, passed in results.items():
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {status}  {test_name}")

print()
print(f"  {pass_count}/{total} tests passed")

all_ok = all(results.values())
print()
if all_ok:
    print("  ✅ ALL TESTS PASSED — backend is aligned with models")
else:
    print("  ⚠️  SOME TESTS FAILED — see details above")

# Cleanup temp fix script
try:
    os.remove("fix_registry.py")
except Exception:
    pass

sys.exit(0 if all_ok else 1)
