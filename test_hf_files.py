#!/usr/bin/env python3
"""Test script to check HuggingFace repository files"""

import os
import certifi

# FIX SSL ISSUE BEFORE ANY IMPORTS
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
os.environ['SSL_CERT_FILE'] = certifi.where()

from huggingface_hub import list_repo_files, hf_hub_download
from pathlib import Path

print("=" * 70)
print("CHECKING HUGGINGFACE REPOSITORIES")
print("=" * 70)
print(f"Using SSL certs from: {certifi.where()}")
print()

print("\n📁 LSTM Model Repository: vlakvindu/Dementia_LSTM_Model")
print("-" * 70)
try:
    files = list_repo_files('vlakvindu/Dementia_LSTM_Model')
    for f in sorted(files):
        print(f"  ✓ {f}")
except Exception as e:
    print(f"  ❌ ERROR: {e}")

print("\n📁 Risk Classifier Repository: vlakvindu/Dementia_Risk_Clasification_model")
print("-" * 70)
try:
    files = list_repo_files('vlakvindu/Dementia_Risk_Clasification_model')
    for f in sorted(files):
        print(f"  ✓ {f}")
except Exception as e:
    print(f"  ❌ ERROR: {e}")

print("\n" + "=" * 70)
print("CHECKING DOWNLOADS")
print("=" * 70)

# Try to download each file
test_dir = Path("test_hf_download")
test_dir.mkdir(exist_ok=True)

repos = [
    ('vlakvindu/Dementia_LSTM_Model', 'lstm_model.keras'),
    ('vlakvindu/Dementia_LSTM_Model', 'lstm_scaler.pkl'),
    ('vlakvindu/Dementia_Risk_Clasification_model', 'risk_logreg.pkl'),
    ('vlakvindu/Dementia_Risk_Clasification_model', 'risk_scaler.pkl'),
    ('vlakvindu/Dementia_Risk_Clasification_model', 'risk_label_encoder.pkl'),
]

for repo, filename in repos:
    try:
        print(f"\n📥 Downloading {filename} from {repo}...")
        path = hf_hub_download(repo_id=repo, filename=filename, local_dir=str(test_dir))
        print(f"  ✓ SUCCESS: {path}")
    except Exception as e:
        print(f"  ❌ FAILED: {str(e)[:100]}")
