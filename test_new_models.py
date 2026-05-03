"""
Test script to verify new models are properly loaded
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_model_loading():
    """Test if models load successfully"""
    print("=" * 70)
    print("🔍 Testing Model Loading")
    print("=" * 70)
    
    try:
        from src.models.game.model_registry import (
            load_all_models, 
            get_risk_classifier,
            get_risk_scaler,
            get_label_encoder,
            get_lstm_model,
            get_lstm_scaler
        )
        
        # Load all models
        print("\n📦 Loading all models...")
        load_all_models()
        
        # Get individual models
        models = {
            'risk_classifier': get_risk_classifier(),
            'scaler': get_risk_scaler(),
            'label_encoder': get_label_encoder(),
            'lstm_model': get_lstm_model(),
            'lstm_scaler': get_lstm_scaler()
        }
        
        print("\n✅ Model Loading Summary:")
        print("-" * 70)
        for model_name, model_obj in models.items():
            status = "✓ Loaded" if model_obj is not None else "✗ Failed"
            print(f"{model_name:20} : {status}")
        
        return models
        
    except Exception as e:
        print(f"\n❌ Error loading models: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_risk_classifier_inference(models):
    """Test risk classifier with sample data"""
    print("\n" + "=" * 70)
    print("🧪 Testing Risk Classifier Inference")
    print("=" * 70)
    
    try:
        import numpy as np
        
        risk_classifier = models.get('risk_classifier')
        scaler = models.get('scaler')
        label_encoder = models.get('label_encoder')
        
        if risk_classifier is None:
            print("❌ Risk classifier not loaded")
            return False
        
        # Create sample features (adjust based on your model's expected features)
        print("\n📊 Creating sample feature data...")
        sample_features = np.array([[
            75.0,   # avg_reaction_time
            3,      # errors_count
            0.85,   # accuracy_rate
            120.0,  # session_duration
            5,      # levels_completed
            2.5     # difficulty_level
        ]])
        
        print(f"Sample features shape: {sample_features.shape}")
        print(f"Sample features: {sample_features[0]}")
        
        # Scale if scaler available
        if scaler is not None:
            print("\n⚙️ Applying feature scaling...")
            sample_features_scaled = scaler.transform(sample_features)
            print(f"Scaled features: {sample_features_scaled[0]}")
        else:
            sample_features_scaled = sample_features
            print("⚠️ No scaler found, using raw features")
        
        # Make prediction
        print("\n🎯 Making prediction...")
        prediction = risk_classifier.predict(sample_features_scaled)
        prediction_proba = risk_classifier.predict_proba(sample_features_scaled)
        
        print(f"Prediction (numeric): {prediction}")
        print(f"Prediction probabilities: {prediction_proba}")
        
        # Decode label if encoder available
        if label_encoder is not None:
            decoded_label = label_encoder.inverse_transform(prediction)
            print(f"Decoded label: {decoded_label[0]}")
        
        print("\n✅ Risk classifier inference successful!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_attributes(models):
    """Check model attributes and metadata"""
    print("\n" + "=" * 70)
    print("🔬 Checking Model Attributes")
    print("=" * 70)
    
    try:
        risk_classifier = models.get('risk_classifier')
        scaler = models.get('scaler')
        label_encoder = models.get('label_encoder')
        
        if risk_classifier:
            print("\n📋 Risk Classifier:")
            print(f"   Type: {type(risk_classifier).__name__}")
            if hasattr(risk_classifier, 'n_features_in_'):
                print(f"   Features expected: {risk_classifier.n_features_in_}")
            if hasattr(risk_classifier, 'classes_'):
                print(f"   Classes: {risk_classifier.classes_}")
            if hasattr(risk_classifier, 'coef_'):
                print(f"   Coefficients shape: {risk_classifier.coef_.shape}")
        
        if scaler:
            print("\n📋 Feature Scaler:")
            print(f"   Type: {type(scaler).__name__}")
            if hasattr(scaler, 'n_features_in_'):
                print(f"   Features: {scaler.n_features_in_}")
            if hasattr(scaler, 'mean_'):
                print(f"   Mean values: {scaler.mean_}")
        
        if label_encoder:
            print("\n📋 Label Encoder:")
            print(f"   Type: {type(label_encoder).__name__}")
            if hasattr(label_encoder, 'classes_'):
                print(f"   Classes: {label_encoder.classes_}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error checking attributes: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("\n🚀 Starting Model Verification Tests\n")
    
    # Test 1: Load models
    models = test_model_loading()
    if models is None:
        print("\n❌ FAILED: Could not load models")
        return
    
    # Test 2: Check attributes
    test_model_attributes(models)
    
    # Test 3: Run inference
    test_risk_classifier_inference(models)
    
    print("\n" + "=" * 70)
    print("✅ All tests completed!")
    print("=" * 70)
    print("\n💡 Next steps:")
    print("   1. If all tests passed, your models are working correctly")
    print("   2. Start your API: python run_api.py")
    print("   3. Test endpoints with actual requests")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
