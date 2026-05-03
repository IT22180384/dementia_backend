"""
Test script to verify game risk classifier models load correctly
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("=" * 70)
print("Testing Game Risk Classifier Model Loading")
print("=" * 70)

try:
    from src.models.game.model_registry import load_all_models, get_risk_classifier, get_risk_scaler, get_label_encoder
    
    print("\n1. Loading all models...")
    load_all_models()
    
    print("\n2. Testing model retrieval...")
    risk_model = get_risk_classifier()
    scaler = get_risk_scaler()
    encoder = get_label_encoder()
    
    print(f"   ✓ Risk Classifier: {type(risk_model).__name__}")
    print(f"   ✓ Feature Scaler: {type(scaler).__name__ if scaler else 'None'}")
    print(f"   ✓ Label Encoder: {type(encoder).__name__ if encoder else 'None'}")
    
    print("\n3. Testing model prediction...")
    if risk_model is not None:
        import numpy as np
        # Create sample features with 14 dimensions to match the trained model
        sample_features = np.random.rand(1, 14)
        
        if scaler is not None:
            sample_features = scaler.transform(sample_features)
        
        prediction = risk_model.predict(sample_features)
        print(f"   ✓ Sample prediction: {prediction[0]}")
        
        if hasattr(risk_model, 'predict_proba'):
            probabilities = risk_model.predict_proba(sample_features)
            print(f"   ✓ Sample probabilities: {probabilities[0]}")
        
        if encoder is not None:
            label = encoder.inverse_transform([prediction[0]])[0]
            print(f"   ✓ Predicted label: {label}")
    
    print("\n" + "=" * 70)
    print("✅ ALL MODELS LOADED SUCCESSFULLY!")
    print("=" * 70)
    print("\nYour models are correctly integrated and ready to use.")
    print("You can now run the backend API server with: python run_api.py")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    print("\n" + "=" * 70)
    print("Please check the error above and fix any issues.")
    print("=" * 70)
