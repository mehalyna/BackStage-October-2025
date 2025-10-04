#!/usr/bin/env python3
"""
Fresh Streamlit Salary Predictor - Version 2
Built specifically for compatibility with regenerated models
"""

import streamlit as st
import pandas as pd
import json
import os
import sys

# Force reload all modules to ensure fresh imports
if 'numpy' in sys.modules:
    del sys.modules['numpy']
if 'joblib' in sys.modules:
    del sys.modules['joblib']
if 'sklearn' in sys.modules:
    del sys.modules['sklearn']

# Now import with fresh modules
try:
    import numpy as np
    import joblib
    from sklearn.pipeline import Pipeline
    IMPORTS_OK = True
    IMPORT_ERROR = None
except Exception as e:
    IMPORTS_OK = False
    IMPORT_ERROR = str(e)

def load_models_safe():
    """Safely load models with detailed error reporting"""
    if not IMPORTS_OK:
        return None, f"Import error: {IMPORT_ERROR}"
    
    models_dir = 'models'
    
    # Check directory exists
    if not os.path.exists(models_dir):
        return None, f"Models directory '{models_dir}' not found"
    
    try:
        # Load models step by step with detailed error reporting
        models = {}
        
        # Step 1: Load Random Forest
        rf_path = os.path.join(models_dir, 'random_forest_model.pkl')
        if not os.path.exists(rf_path):
            return None, f"Random Forest model file not found: {rf_path}"
        models['rf_model'] = joblib.load(rf_path)
        
        # Step 2: Load Linear Regression
        lr_path = os.path.join(models_dir, 'linear_regression_model.pkl')
        if not os.path.exists(lr_path):
            return None, f"Linear Regression model file not found: {lr_path}"
        models['lr_model'] = joblib.load(lr_path)
        
        # Step 3: Load metadata
        meta_path = os.path.join(models_dir, 'model_metadata.json')
        if not os.path.exists(meta_path):
            return None, f"Metadata file not found: {meta_path}"
        with open(meta_path, 'r') as f:
            models['metadata'] = json.load(f)
        
        return models, None
        
    except Exception as e:
        return None, f"Error loading models: {str(e)} | Type: {type(e).__name__}"

def create_prediction_data(job_title, seniority):
    """Create data for prediction"""
    # Simple feature creation matching training data
    data = {
        'Rating': 4.1 if seniority == 'Middle' else (3.8 if seniority == 'Junior' else 4.3),
        'Founded': 2005 if seniority == 'Middle' else (2010 if seniority == 'Junior' else 2000),
        'hourly': 0,
        'employer_provided': 1,
        'Job Title': job_title,
        'Job Description': job_title
    }
    
    return pd.DataFrame([data])

def main():
    st.set_page_config(
        page_title="AI Salary Predictor v2",
        page_icon="💰",
        layout="wide"
    )
    
    st.title("💰 AI Salary Predictor v2")
    st.subheader("Predict IT salaries using Machine Learning")
    
    # Load models
    with st.spinner("Loading models..."):
        models, error = load_models_safe()
    
    if error:
        st.error(f"❌ {error}")
        st.info("🔧 Models were regenerated. Try refreshing the page.")
        
        # Debug information
        with st.expander("🔍 Debug Information"):
            st.write(f"**Python Version:** {sys.version}")
            st.write(f"**Working Directory:** {os.getcwd()}")
            st.write(f"**Import Status:** {'✅ Success' if IMPORTS_OK else '❌ Failed'}")
            if not IMPORTS_OK:
                st.write(f"**Import Error:** {IMPORT_ERROR}")
            
            # Check models directory
            models_dir = 'models'
            if os.path.exists(models_dir):
                files = os.listdir(models_dir)
                st.write(f"**Models Directory:** ✅ Found {len(files)} files")
                for file in sorted(files):
                    st.write(f"  - {file}")
            else:
                st.write("**Models Directory:** ❌ Not found")
        
        return
    
    st.success("✅ Models loaded successfully!")
    
    # Show model information
    metadata = models['metadata']
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Dataset Size", f"{metadata['n_samples']} samples")
    with col2:
        st.metric("Features", metadata['n_features'])
    with col3:
        st.metric("Job Titles", len(metadata.get('job_titles', [])))
    
    # Prediction interface
    st.subheader("🎯 Make a Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Job title selection from actual training data
        job_titles = metadata.get('job_titles', ['Data Scientist', 'Data Engineer', 'Data Analyst'])
        job_title = st.selectbox("Job Title:", job_titles[:12])  # Limit to first 12
        
        seniority = st.selectbox("Seniority Level:", ["Junior", "Middle", "Senior"])
        
        predict_button = st.button("🔮 Predict Salary", type="primary")
    
    with col2:
        if predict_button:
            with st.spinner("Making prediction..."):
                try:
                    # Create prediction data
                    pred_data = create_prediction_data(job_title, seniority)
                    
                    # Make predictions
                    rf_pred = models['rf_model'].predict(pred_data)[0]
                    lr_pred = models['lr_model'].predict(pred_data)[0]
                    avg_pred = (rf_pred + lr_pred) / 2
                    
                    # Display results
                    st.subheader("💰 Salary Predictions")
                    
                    col_rf, col_lr, col_avg = st.columns(3)
                    with col_rf:
                        st.metric("Random Forest", f"${rf_pred:,.0f}")
                    with col_lr:
                        st.metric("Linear Regression", f"${lr_pred:,.0f}")
                    with col_avg:
                        st.metric("Average", f"${avg_pred:,.0f}")
                    
                    st.info(f"Prediction for {seniority} {job_title}")
                    
                    # Model performance
                    st.subheader("📊 Model Performance")
                    rf_r2 = metadata.get('rf_r2', 0)
                    lr_r2 = metadata.get('lr_r2', 0)
                    
                    col_perf1, col_perf2 = st.columns(2)
                    with col_perf1:
                        st.metric("Random Forest R²", f"{rf_r2:.3f}")
                    with col_perf2:
                        st.metric("Linear Regression R²", f"{lr_r2:.3f}")
                    
                except Exception as e:
                    st.error(f"❌ Prediction error: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("*Built with Streamlit and scikit-learn*")

if __name__ == "__main__":
    main()