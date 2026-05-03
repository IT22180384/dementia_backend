import requests
import json
import uuid
import datetime

# The API endpoint where your FastAPI server is running
API_URL = "http://127.0.0.1:8000/game/session"

# Generate a random test user and session ID
test_user_id = str(uuid.uuid4())
test_session_id = str(uuid.uuid4())

print("=================================================================")
print("🎮 DEMENTIA GAME BACKEND TEST: Risk Prediction API")
print("=================================================================")
print(f"Creating test session for new user: {test_user_id[:8]}...")

# Example 1: A player who is struggling (High errors, very slow reaction time)
# This simulates 5 trials of a game where the user has low accuracy and high RT
payload = {
    "userId": test_user_id,
    "sessionId": test_session_id,
    "gameType": "card_matching",
    "level": 1,
    "trials": [
        {"rt_raw": 3.5, "correct": 0, "error": 1, "hint_used": 1},
        {"rt_raw": 4.1, "correct": 0, "error": 1, "hint_used": 0},
        {"rt_raw": 2.8, "correct": 1, "error": 0, "hint_used": 0},
        {"rt_raw": 5.0, "correct": 0, "error": 1, "hint_used": 1},
        {"rt_raw": 3.2, "correct": 1, "error": 0, "hint_used": 0}
    ]
}

print(f"\n📡 SENDING GAME DATA (5 trails, 2/5 correct, slow ~3.5s RT)...")

try:
    # Send the POST request to your API
    response = requests.post(API_URL, json=payload, headers={"Content-Type": "application/json"})
    
    if response.status_code == 201 or response.status_code == 200:
        result = response.json()
        print("\n✅ API RESPONDED SUCCESSFULLY!\n")
        
        # Parse the output
        features = result.get("features", {})
        prediction = result.get("prediction", {})
        risk_level = prediction.get("riskLevel", "UNKNOWN")
        risk_score = prediction.get("riskScore0_100", 0.0)
        
        print("🧠 COMPUTED COGNITIVE FEATURES:")
        print(f"  Accuracy:    {features.get('accuracy', 0)*100:.0f}%")
        print(f"  SAC Score:   {features.get('sac', 0):.4f} (Speed-Accuracy Composite)")
        print(f"  IES Score:   {features.get('ies', 0):.4f} (Inverse Efficiency)")
        print(f"  Adj RT:      {features.get('rtAdjMedian', 0):.2f}s (Motor-adjusted reaction time)")
        
        print("\n🚨 MODEL RISK PREDICTION RESULT:")
        print(f"  Risk Level:  {risk_level}")
        print(f"  Risk Score:  {risk_score}/100")
        
        probs = prediction.get("riskProbability", {})
        print(f"  Confidence:  High={probs.get('HIGH',0)*100:.1f}% | Med={probs.get('MEDIUM',0)*100:.1f}% | Low={probs.get('LOW',0)*100:.1f}%")
        print("\n=================================================================")
        print("Test Passed: The Game API and Logistic Regression model are connected and working!")
        
    else:
        print(f"\n❌ API ERROR (Status Code: {response.status_code})")
        print(response.text)

except requests.exceptions.ConnectionError:
    print("\n❌ CONNECTION ERROR: Could not connect to the API.")
    print("Is your FastAPI server running? Make sure 'python run_api.py' is running in another terminal.")

except Exception as e:
    print(f"\n❌ UNEXPECTED ERROR: {e}")
