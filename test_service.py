from src.services.game_service import predict_risk

result = predict_risk(
    sessions=[],
    current_features={
        'sac': 0.039,
        'ies': 15.025,
        'accuracy': 0.06,
        'rtAdjMedian': 1.502,
        'variability': 0.164
    },
    lstm_score=0.0
)

print(f"\n✅ AFTER INVERSION FIX:")
print(f"   Risk Level: {result['riskLevel']}")
print(f"   Score: {result['riskScore0_100']}/100")
print(f"   Probabilities: {result['riskProbability']}")
