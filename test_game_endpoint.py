#!/usr/bin/env python3
"""Test game session endpoint after SSL fix"""

import requests
import json
import time

# Wait a bit for models to load
print("Waiting for models to load...")
time.sleep(5)

# Test payload
test_payload = {
    "userId": "test-user-ssl-fix-123",
    "trials": 10,
    "responses": [
        {"trial": 1, "reactionTime": 0.45, "correct": True},
        {"trial": 2, "reactionTime": 0.52, "correct": True},
        {"trial": 3, "reactionTime": 0.48, "correct": False},
        {"trial": 4, "reactionTime": 0.61, "correct": True},
        {"trial": 5, "reactionTime": 0.55, "correct": True},
        {"trial": 6, "reactionTime": 0.49, "correct": True},
        {"trial": 7, "reactionTime": 0.58, "correct": True},
        {"trial": 8, "reactionTime": 0.44, "correct": True},
        {"trial": 9, "reactionTime": 0.53, "correct": False},
        {"trial": 10, "reactionTime": 0.50, "correct": True},
    ]
}

print("\n" + "=" * 70)
print("TESTING GAME SESSION ENDPOINT")
print("=" * 70)
print(f"Endpoint: http://localhost:8080/game/session")
print(f"Method: POST")
print(f"User: {test_payload['userId']}")
print(f"Trials: {test_payload['trials']}")

try:
    response = requests.post(
        "http://localhost:8080/game/session",
        json=test_payload,
        headers={"Content-Type": "application/json"},
        timeout=30
    )
    
    print(f"\nStatus Code: {response.status_code}")
    print("-" * 70)
    
    if response.status_code == 200:
        result = response.json()
        print("✅ SUCCESS! Response:")
        print(json.dumps(result, indent=2))
        
        # Extract key info
        if "risk_prediction" in result:
            print(f"\n🎯 Risk Prediction: {result['risk_prediction']}")
        if "accuracy" in result:
            print(f"📊 Accuracy: {result['accuracy']}%")
        if "lstm_decline_score" in result:
            print(f"📈 LSTM Decline Score: {result['lstm_decline_score']}")
            
    else:
        print(f"❌ ERROR! Status: {response.status_code}")
        print(f"Response: {response.text}")
        
except Exception as e:
    print(f"❌ Connection Error: {e}")
    print("Make sure the API server is running on http://localhost:8080")

print("\n" + "=" * 70)
