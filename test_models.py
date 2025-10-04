#!/usr/bin/env python3
"""
Test script to verify models can be loaded properly
"""
import sys
import os

try:
    import numpy as np
    print(f"✅ NumPy version: {np.__version__}")
except ImportError as e:
    print(f"❌ NumPy import error: {e}")
    sys.exit(1)

try:
    import sklearn
    print(f"✅ Scikit-learn version: {sklearn.__version__}")
except ImportError as e:
    print(f"❌ Scikit-learn import error: {e}")
    sys.exit(1)

try:
    import joblib
    print(f"✅ Joblib version: {joblib.__version__}")
except ImportError as e:
    print(f"❌ Joblib import error: {e}")
    sys.exit(1)

# Test model loading
models_dir = 'models'
if not os.path.exists(models_dir):
    print(f"❌ Models directory '{models_dir}' not found!")
    sys.exit(1)

required_files = [
    'linear_regression_model.pkl',
    'random_forest_model.pkl', 
    'preprocessor.pkl',
    'kmeans_model.pkl',
    'pca_model.pkl',
    'model_metadata.json'
]

print(f"\n📁 Checking models directory: {models_dir}")
for file in required_files:
    filepath = os.path.join(models_dir, file)
    if os.path.exists(filepath):
        print(f"✅ {file}")
    else:
        print(f"❌ {file} - MISSING")

# Try loading one model
try:
    import joblib
    model_path = os.path.join(models_dir, 'random_forest_model.pkl')
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print(f"✅ Successfully loaded random forest model")
        print(f"   Model type: {type(model)}")
    else:
        print(f"❌ Cannot test model loading - file not found")
except Exception as e:
    print(f"❌ Error loading model: {e}")

print(f"\n🎯 Test completed!")