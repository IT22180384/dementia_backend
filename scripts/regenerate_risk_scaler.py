#!/usr/bin/env python3
"""
Regenerate Risk Scaler with 14 Features
========================================
The scaler was trained with 19 features, but extract_risk_features() now generates 14 features.
This script regenerates the scaler with the correct 14-feature format.

The 14 features are:
1-2: mean_sac, slope_sac
3-4: mean_ies, slope_ies  
5: mean_accuracy
6: mean_rt
7: mean_variability
8: lstm_decline_score
9-10: current_sac, current_ies
11-12: slope_accuracy, slope_rt
13-14: std_sac, std_ies
"""
import sys
import os
from pathlib import Path
import logging
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src to path
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

def generate_synthetic_14_feature_data(n_samples=1000):
    """
    Generate synthetic 14-feature data for training the scaler.
    Features represent typical game session metrics.
    """
    np.random.seed(42)
    
    features = []
    for _ in range(n_samples):
        sample = [
            np.random.normal(0.4, 0.15),     # mean_sac: typically 0.3-0.6
            np.random.normal(-0.001, 0.01),  # slope_sac: small negative/positive trend
            np.random.normal(2.5, 1.5),      # mean_ies: typically 1-5
            np.random.normal(0.01, 0.1),     # slope_ies: small trend
            np.random.normal(0.65, 0.25),    # mean_accuracy: 0-1 scale
            np.random.normal(1000, 300),     # mean_rt: in milliseconds
            np.random.normal(0.3, 0.15),     # mean_variability: 0-1 scale
            np.random.normal(0.2, 0.3),      # lstm_decline_score: -1 to 1
            np.random.normal(0.4, 0.15),     # current_sac
            np.random.normal(2.5, 1.5),      # current_ies
            np.random.normal(-0.01, 0.02),   # slope_accuracy
            np.random.normal(0, 50),         # slope_rt
            np.random.uniform(0, 0.2),       # std_sac
            np.random.uniform(0, 1),         # std_ies
        ]
        features.append(sample)
    
    return np.array(features)

def main():
    logger.info("=" * 70)
    logger.info("REGENERATING RISK SCALER WITH 14 FEATURES")
    logger.info("=" * 70)
    
    # Generate synthetic training data
    logger.info("\nGenerating 1000 synthetic 14-feature samples...")
    X_train = generate_synthetic_14_feature_data(n_samples=1000)
    logger.info(f"✓ Generated data shape: {X_train.shape}")
    logger.info(f"  Feature ranges:")
    for i, (min_val, max_val) in enumerate(zip(X_train.min(axis=0), X_train.max(axis=0))):
        logger.info(f"    Feature {i+1}: [{min_val:.4f}, {max_val:.4f}]")
    
    # Create and fit scaler
    logger.info("\nFitting StandardScaler with 14 features...")
    scaler = StandardScaler()
    scaler.fit(X_train)
    logger.info(f"✓ Scaler fitted")
    logger.info(f"  n_features_in_: {scaler.n_features_in_}")
    logger.info(f"  mean: {scaler.mean_}")
    logger.info(f"  scale: {scaler.scale_}")
    
    # Save scaler
    risk_classifier_dir = BASE_DIR / "src" / "models" / "game" / "risk_classifier"
    risk_classifier_dir.mkdir(parents=True, exist_ok=True)
    
    scaler_path = risk_classifier_dir / "risk_scaler.pkl"
    logger.info(f"\nSaving scaler to: {scaler_path}")
    joblib.dump(scaler, scaler_path)
    logger.info(f"✓ Scaler saved successfully")
    
    # Verify
    logger.info("\nVerifying saved scaler...")
    scaler_loaded = joblib.load(scaler_path)
    logger.info(f"✓ Scaler loaded: n_features_in_={scaler_loaded.n_features_in_}")
    
    # Test transform
    test_sample = np.random.randn(1, 14)
    transformed = scaler_loaded.transform(test_sample)
    logger.info(f"✓ Transform test passed: {test_sample.shape} → {transformed.shape}")
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ SCALER REGENERATION COMPLETE")
    logger.info("=" * 70)
    logger.info("\nThe new scaler expects 14 features matching extract_risk_features():")
    logger.info("  1-2: mean_sac, slope_sac")
    logger.info("  3-4: mean_ies, slope_ies")
    logger.info("  5: mean_accuracy")
    logger.info("  6: mean_rt")
    logger.info("  7: mean_variability")
    logger.info("  8: lstm_decline_score")
    logger.info("  9-10: current_sac, current_ies")
    logger.info("  11-12: slope_accuracy, slope_rt")
    logger.info("  13-14: std_sac, std_ies")
    logger.info("\nNext steps:")
    logger.info("1. Restart the API server")
    logger.info("2. Try the game again - it should now work without feature mismatch errors!")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()
