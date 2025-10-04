#!/usr/bin/env python3
"""
Test script to verify models work correctly
"""

import sys
import traceback

def test_models():
    try:
        print("🔍 Testing model loading and prediction...")
        
        # Import required libraries
        import joblib
        import pandas as pd
        import json
        import os
        
        print(f"✓ Python version: {sys.version}")
        print(f"✓ Working directory: {os.getcwd()}")
        
        # Check if models directory exists
        models_dir = "models"
        if not os.path.exists(models_dir):
            print(f"❌ Models directory '{models_dir}' not found!")
            return False
            
        # List files in models directory
        model_files = os.listdir(models_dir)
        print(f"✓ Found {len(model_files)} files in models directory:")
        for file in sorted(model_files):
            print(f"  - {file}")
            
        # Load models
        print("\n🤖 Loading models...")
        rf_model = joblib.load(os.path.join(models_dir, 'random_forest_model.pkl'))
        lr_model = joblib.load(os.path.join(models_dir, 'linear_regression_model.pkl'))
        
        # Load metadata
        with open(os.path.join(models_dir, 'model_metadata.json'), 'r') as f:
            metadata = json.load(f)
            
        print("✓ All models loaded successfully!")
        print(f"✓ Feature columns: {metadata['feature_columns']}")
        
        # Test prediction
        print("\n🧪 Testing prediction...")
        test_data = {
            'Rating': 4.2,
            'Founded': 2010,
            'hourly': 0,
            'employer_provided': 1,
            'Job Title': 'Data Scientist',
            'Job Description': 'Data Scientist'
        }
        
        test_df = pd.DataFrame([test_data])
        print(f"✓ Test data: {test_data}")
        
        prediction_rf = rf_model.predict(test_df)[0]
        prediction_lr = lr_model.predict(test_df)[0]
        
        print(f"✓ Random Forest prediction: ${prediction_rf:,.0f}")
        print(f"✓ Linear Regression prediction: ${prediction_lr:,.0f}")
        
        print("\n🎉 All tests passed! Models are working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        print("\n📋 Full traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_models()
    sys.exit(0 if success else 1)