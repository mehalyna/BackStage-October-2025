import streamlit as st
import sys
import os

# Debug info
st.sidebar.write("🔧 Debug Info")
st.sidebar.write(f"Python version: {sys.version}")
st.sidebar.write(f"Working directory: {os.getcwd()}")

# Add current directory to path
sys.path.insert(0, os.getcwd())

def check_imports():
    """Check all required imports"""
    try:
        import numpy as np
        st.sidebar.success(f"✅ NumPy {np.__version__}")
    except Exception as e:
        st.sidebar.error(f"❌ NumPy: {e}")
        return False
    
    try:
        import sklearn
        st.sidebar.success(f"✅ Scikit-learn {sklearn.__version__}")
    except Exception as e:
        st.sidebar.error(f"❌ Scikit-learn: {e}")
        return False
    
    try:
        import joblib
        st.sidebar.success(f"✅ Joblib {joblib.__version__}")
    except Exception as e:
        st.sidebar.error(f"❌ Joblib: {e}")
        return False
    
    try:
        import pandas as pd
        st.sidebar.success(f"✅ Pandas {pd.__version__}")
    except Exception as e:
        st.sidebar.error(f"❌ Pandas: {e}")
        return False
    
    return True

def check_models():
    """Check if model files exist"""
    models_dir = 'models'
    required_files = [
        'linear_regression_model.pkl',
        'random_forest_model.pkl', 
        'preprocessor.pkl',
        'kmeans_model.pkl',
        'pca_model.pkl',
        'model_metadata.json'
    ]
    
    if not os.path.exists(models_dir):
        st.sidebar.error(f"❌ Models directory not found")
        return False
    
    missing_files = []
    for file in required_files:
        filepath = os.path.join(models_dir, file)
        if os.path.exists(filepath):
            st.sidebar.success(f"✅ {file}")
        else:
            st.sidebar.error(f"❌ {file}")
            missing_files.append(file)
    
    return len(missing_files) == 0

def test_model_loading():
    """Test loading a model"""
    try:
        import joblib
        model_path = os.path.join('models', 'random_forest_model.pkl')
        model = joblib.load(model_path)
        st.sidebar.success("✅ Model loading test passed")
        return True
    except Exception as e:
        st.sidebar.error(f"❌ Model loading failed: {e}")
        return False

# Main app
st.title("🔧 Streamlit App Diagnostics")

st.write("This diagnostic page will help identify and fix the numpy_core issue.")

# Run all checks
st.subheader("🧪 Running Diagnostics...")

if check_imports():
    st.success("✅ All required packages imported successfully")
else:
    st.error("❌ Package import issues detected")
    st.stop()

if check_models():
    st.success("✅ All model files found")
else:
    st.error("❌ Model files missing")
    st.info("Please run the Jupyter notebook to generate model files")
    st.stop()

if test_model_loading():
    st.success("✅ Model loading test passed")
    st.balloons()
    
    # If everything works, show a button to launch the main app
    if st.button("🚀 Launch Main Salary Predictor App"):
        st.write("All tests passed! You can now run the main app:")
        st.code("streamlit run app.py")
        
else:
    st.error("❌ Model loading failed")
    
st.subheader("📋 Troubleshooting Steps")
st.info("""
1. **Check Python Environment**: Ensure you're using the correct Python version
2. **Install Dependencies**: Run `pip install -r requirements.txt`
3. **Generate Models**: Run the Jupyter notebook `salary_demo_colab.ipynb`
4. **Clear Cache**: In Streamlit, press 'C' to clear cache
5. **Restart App**: Stop and restart the Streamlit app
""")

# Add option to clear Streamlit cache
if st.button("🗑️ Clear Streamlit Cache"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.success("Cache cleared! Please refresh the page.")
    st.experimental_rerun()