# � BackStage October 2025 - ML & Data Science Demo

A comprehensive machine learning project featuring **AI Salary Prediction** and **Live Web Scraping** capabilities. This repository showcases end-to-end ML workflows from data collection to production-ready web applications.

## 📦 Project Components

### 1. 💰 AI Salary Predictor (Streamlit App)
A machine learning-powered web application for predicting IT salaries based on job titles and seniority levels.

### 2. 🌐 Live Web Scraping (Jupyter Notebook)
Real-time web scraping and word cloud generation from Hacker News job postings.

### 3. 📊 ML Pipeline (Jupyter Notebook)
Complete machine learning pipeline with data preprocessing, model training, and evaluation.

## 🚀 Features

### AI Salary Predictor
- **Interactive Salary Prediction**: Real-time salary estimates for data science roles
- **Multiple ML Models**: Random Forest (94.9% accuracy) and Linear Regression (86.9% accuracy)
- **Seniority Adjustments**: Junior, Middle, Senior level predictions
- **Model Performance Metrics**: R² scores, RMSE, and comparison charts
- **742+ Training Samples**: Based on real Glassdoor salary data

### Live Web Scraping
- **Real-time Data Collection**: Scrapes Hacker News "Who's Hiring" posts
- **Word Cloud Generation**: Visual representation of trending job keywords
- **Natural Language Processing**: NLTK-powered text analysis
- **Error Handling**: Robust scraping with timeout management

## 📋 Prerequisites & Setup

### Python Environment Requirements
- **Python 3.11+** (recommended for compatibility)
- **Required packages**: See `requirements.txt`

### Quick Start (Option 1 - Use Fresh App)
```bash
# Clone the repository
git clone <repository-url>
cd BackStage-October-2025

# Install dependencies
pip install -r requirements.txt

# Generate ML models (required first time)
python regenerate_models.py

# Run the latest version of the app
streamlit run fresh_app.py --server.port 8506
```

### Alternative Setup (Option 2 - Jupyter First)
```bash
# Install dependencies
pip install -r requirements.txt

# Open and run the ML notebook to generate models
jupyter notebook salary_demo_colab.ipynb
# Run all cells to generate the models/ directory

# Run the Streamlit app
streamlit run simple_app.py
```

### Files You'll Need
After running the setup, you should have:
- `models/` directory with:
  - `linear_regression_model.pkl`
  - `random_forest_model.pkl` 
  - `preprocessor.pkl`
  - `kmeans_model.pkl`
  - `pca_model.pkl`
  - `model_metadata.json`
streamlit run app.py
```

## 🎯 How to Use

### AI Salary Predictor
1. **Start the Application**:
   ```bash
   streamlit run fresh_app.py --server.port 8506
   ```

2. **Open in Browser**: Navigate to `http://localhost:8506`

3. **Make Predictions**:
   - Select your **Job Title** from the dropdown (Data Scientist, Data Engineer, etc.)
   - Choose your **Seniority Level** (Junior/Middle/Senior)
   - Click **"� Predict Salary"**

4. **View Results**:
   - **Random Forest Prediction**: High-accuracy model (94.9% R²)
   - **Linear Regression Prediction**: Interpretable baseline (86.9% R²)
   - **Average Prediction**: Combined estimate
   - **Model Performance Metrics**: Real-time accuracy indicators

### Live Web Scraping
1. **Open Jupyter Notebook**:
   ```bash
   jupyter notebook live_scrape.ipynb
   ```

2. **Run Cells Sequentially**:
   - Import libraries and setup
   - Scrape Hacker News "Who's Hiring" posts
   - Generate word cloud visualization
   - View trending job keywords

## 📊 Project Structure

```
BackStage-October-2025/
├── 🎯 Core Applications
│   ├── fresh_app.py                    # Latest Streamlit app (recommended)
│   ├── simple_app.py                   # Alternative Streamlit app
│   └── app.py                          # Original Streamlit app
│
├── 🤖 Machine Learning
│   ├── salary_demo_colab.ipynb         # Complete ML pipeline
│   ├── regenerate_models.py            # Model generation script
│   └── test_models_simple.py           # Model validation
│
├── 🌐 Web Scraping
│   └── live_scrape.ipynb               # Live Hacker News scraping
│
├── 📊 Data
│   ├── glassdoor_jobs.csv              # Primary dataset (956 records)
│   ├── salary_data_cleaned.csv         # Processed salary data
│   └── eda_data.csv                    # Exploratory data analysis
│
├── 🎯 Generated Models
│   └── models/                         # ML models (generated)
│       ├── linear_regression_model.pkl
│       ├── random_forest_model.pkl
│       ├── preprocessor.pkl
│       ├── kmeans_model.pkl
│       ├── pca_model.pkl
│       └── model_metadata.json
│
└── 📝 Documentation
    ├── README.md                       # This file
    ├── requirements.txt                # Dependencies
    └── examples.md                     # Usage examples
```

## 🔧 Technical Details

### Machine Learning Pipeline
- **Dataset**: 956 Glassdoor job records → 742 clean salary records
- **Salary Range**: $13,500 - $254,000 annual compensation
- **Feature Engineering**: Rating, Founded year, Job Title, Job Description
- **Models**:
  - **Random Forest**: 94.9% R² accuracy, handles non-linear relationships
  - **Linear Regression**: 86.9% R² accuracy, interpretable baseline

### Supported Job Titles (From Training Data)
- Data Scientist (131 samples)
- Data Engineer (53 samples)  
- Senior Data Scientist (34 samples)
- Data Analyst (15 samples)
- Senior Data Engineer (14 samples)
- Machine Learning Engineer
- Product Manager
- Marketing Data Analyst
- Research Scientist
- And more...

### Web Scraping Features
- **Source**: Hacker News "Who's Hiring" monthly posts
- **Libraries**: requests, BeautifulSoup4, NLTK
- **Processing**: STOPWORDS filtering, word frequency analysis
- **Output**: matplotlib word clouds, trending keywords
- **Error Handling**: Timeout management, graceful failures

### Environment Compatibility
- **Tested with**: Python 3.11.4, 3.12.10, 3.13.3
- **NumPy**: Compatible with versions 1.24.3 - 2.3.3
- **Scikit-learn**: 1.6.1 - 1.7.2 (models auto-adapt)
- **Key Dependencies**: streamlit, pandas, joblib, matplotlib

## 📈 Model Performance & Results

### Salary Prediction Accuracy
- **Random Forest Model**: 94.9% R² (explains 94.9% of salary variance)
- **Linear Regression Model**: 86.9% R² (interpretable baseline)
- **Training Data**: 742 salary records from Glassdoor
- **Salary Range**: $13,500 - $254,000 annual compensation

### Example Predictions
```
Job Title: Data Scientist (Senior)
Random Forest: $145,000
Linear Regression: $142,000
Average: $143,500

Job Title: Data Engineer (Middle) 
Random Forest: $86,846
Linear Regression: $106,197
Average: $96,522
```

### Dataset Statistics
- **Total Records**: 956 job postings
- **Clean Salary Data**: 742 records (77.6% success rate)
- **Top Job Titles**:
  - Data Scientist: 131 records (17.7%)
  - Data Engineer: 53 records (7.1%) 
  - Senior Data Scientist: 34 records (4.6%)

## 🚀 Deployment Options

### Local Development
```bash
# Quick start
streamlit run fresh_app.py --server.port 8506

# With model regeneration
python regenerate_models.py && streamlit run fresh_app.py
```

### Streamlit Cloud Deployment
1. Push repository to GitHub
2. Connect to [share.streamlit.io](https://share.streamlit.io)
3. Deploy with one-click
4. Set main file to `fresh_app.py`

### Docker Deployment
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN python regenerate_models.py

EXPOSE 8501
CMD ["streamlit", "run", "fresh_app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
```

### Production Considerations
- **Model Updates**: Re-run `regenerate_models.py` with new data
- **Scaling**: Use Streamlit Cloud or containerization for multiple users
- **Monitoring**: Add logging for prediction requests and errors
- **Security**: Validate inputs and sanitize file uploads if added

## 🐛 Troubleshooting

### Common Issues & Solutions

#### 1. **"Error loading models: No module named 'numpy._core'"**
**Cause**: NumPy version mismatch between model training and loading environments.

**Solution**:
```bash
# Regenerate models with current environment
python regenerate_models.py

# Or use the Python version that Streamlit uses
C:\Users\<username>\AppData\Local\Programs\Python\Python311\python.exe regenerate_models.py

# Then restart Streamlit
streamlit run fresh_app.py --server.port 8506
```

#### 2. **Scikit-learn Version Warnings**
**Cause**: Models trained with different scikit-learn version.

**Solution**:
```bash
# Check current version
python -c "import sklearn; print(sklearn.__version__)"

# Regenerate models to match current environment
python regenerate_models.py
```

#### 3. **Import Errors or Missing Dependencies**
**Solution**:
```bash
# Install all required packages
pip install -r requirements.txt

# Or install specific packages
pip install streamlit pandas scikit-learn joblib matplotlib seaborn
```

#### 4. **Port Already in Use**
**Solution**:
```bash
# Use different port
streamlit run fresh_app.py --server.port 8507

# Or kill existing processes
taskkill /F /IM streamlit.exe  # Windows
# pkill -f streamlit           # Linux/Mac
```

#### 5. **Models Directory Not Found**
**Solution**:
```bash
# Generate models first
python regenerate_models.py

# Or run the Jupyter notebook
jupyter notebook salary_demo_colab.ipynb
# Execute all cells to create models/ directory
```

#### 6. **Web Scraping Timeout Errors**
**Solution**: The live scraping notebook includes timeout handling. If scraping fails:
- Check internet connection
- Try running cells individually
- Hacker News might be temporarily unavailable

### Environment-Specific Notes

#### Python Version Compatibility
- **Recommended**: Python 3.11+ for best compatibility
- **Tested**: Works with Python 3.11.4, 3.12.10, 3.13.3
- **Issue**: Different Python versions may have different NumPy versions

#### Windows-Specific
- Use `taskkill /F /IM streamlit.exe` to stop Streamlit processes
- PowerShell: `Get-Command python` to find Python installations
- Path format: `C:\Users\<username>\AppData\Local\Programs\Python\Python311\python.exe`

#### Model Regeneration
If you encounter persistent compatibility issues:
1. Identify your Python version: `python --version`
2. Regenerate models with that exact Python: `python regenerate_models.py`
3. Test model loading: `python test_models_simple.py`
4. Restart Streamlit app

## 📁 Detailed File Structure

```
BackStage-October-2025/
├── 🎯 Streamlit Applications
│   ├── fresh_app.py                    # ⭐ RECOMMENDED - Latest version with compatibility fixes
│   ├── simple_app.py                   # Alternative version with basic features  
│   ├── app.py                          # Original version with full features
│   └── debug_app.py                    # Diagnostic tool for troubleshooting
│
├── 🤖 Machine Learning & Data Processing
│   ├── salary_demo_colab.ipynb         # 📊 Complete ML pipeline (training & evaluation)
│   ├── regenerate_models.py            # 🔧 Model generation script (fixes compatibility)
│   ├── test_models_simple.py           # ✅ Model validation and testing
│   └── test_models.py                  # Additional model testing utilities
│
├── 🌐 Web Scraping & Data Collection
│   └── live_scrape.ipynb               # 🔴 LIVE - Hacker News scraping & word clouds
│
├── 📊 Datasets
│   ├── glassdoor_jobs.csv              # 🎯 Primary dataset (956 job records)
│   ├── salary_data_cleaned.csv         # Processed salary data
│   └── eda_data.csv                    # Exploratory data analysis results
│
├── 🎯 Generated Assets
│   ├── models/                         # 🤖 ML models (auto-generated)
│   │   ├── linear_regression_model.pkl
│   │   ├── random_forest_model.pkl
│   │   ├── preprocessor.pkl
│   │   ├── kmeans_model.pkl
│   │   ├── pca_model.pkl
│   │   └── model_metadata.json
│   └── ai_demo_outputs/                # Sample prediction outputs
│
└── � Documentation & Configuration
    ├── README.md                       # 📖 This comprehensive guide
    ├── requirements.txt                # 📦 Python dependencies
    ├── examples.md                     # 💡 Usage examples
    └── prompts.txt                     # 🎯 Development prompts
```

## 🔄 Development Workflow

### For First-Time Setup
```bash
# 1. Clone and install
git clone <repository-url>
cd BackStage-October-2025
pip install -r requirements.txt

# 2. Generate ML models
python regenerate_models.py

# 3. Run the app
streamlit run fresh_app.py --server.port 8506

# 4. (Optional) Explore web scraping
jupyter notebook live_scrape.ipynb
```

### For Model Updates
```bash
# 1. Update data (replace glassdoor_jobs.csv with new data)
# 2. Regenerate models
python regenerate_models.py

# 3. Test models
python test_models_simple.py

# 4. Restart app
streamlit run fresh_app.py --server.port 8506
```

### For Development & Debugging
```bash
# Test model compatibility
python test_models_simple.py

# Run diagnostic app
streamlit run debug_app.py

# Check environment
python -c "import sys; print(f'Python: {sys.version}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import sklearn; print(f'Scikit-learn: {sklearn.__version__}')"
```

## � Key Features & Highlights

### 🎯 AI Salary Predictor
- **High Accuracy**: 94.9% R² with Random Forest model
- **Real-world Data**: 742 Glassdoor salary records
- **Multiple Models**: Compare Random Forest vs Linear Regression
- **Seniority Scaling**: Automatic adjustments for experience level
- **Interactive UI**: Clean Streamlit interface with instant predictions

### 🌐 Live Web Scraping
- **Real-time Data**: Scrapes current Hacker News job postings
- **Word Cloud Generation**: Visual trending keywords analysis  
- **NLP Processing**: NLTK-powered text processing and filtering
- **Robust Error Handling**: Timeout management and graceful failures

### 🔧 Technical Excellence
- **Environment Compatibility**: Auto-adapts to Python 3.11-3.13
- **Version Management**: Handles NumPy/scikit-learn compatibility issues
- **Model Persistence**: Joblib serialization with metadata tracking
- **Debugging Tools**: Comprehensive troubleshooting and validation scripts

## 📞 Support & Contributing

### Getting Help
1. **Check Troubleshooting Section**: Common issues and solutions above
2. **Run Diagnostics**: Use `python test_models_simple.py` to validate setup
3. **Environment Issues**: Use `regenerate_models.py` to fix compatibility
4. **Debug Mode**: Run `streamlit run debug_app.py` for detailed error info

### Contributing
1. Fork the repository
2. Create a feature branch
3. Test your changes with `python test_models_simple.py`
4. Ensure models regenerate successfully
5. Submit a pull request

### Development Notes
- **Model Updates**: Always run `regenerate_models.py` after data changes
- **Environment Testing**: Test with multiple Python versions if possible
- **Documentation**: Update README.md for new features
- **Compatibility**: Ensure compatibility with common Python environments

## 🏆 Project Achievements

### ✅ Successfully Resolved Challenges
- **NumPy Compatibility**: Solved `numpy._core` module errors across Python versions
- **Environment Isolation**: Handled multiple Python installations (3.11, 3.12, 3.13)
- **Scikit-learn Versioning**: Auto-adaptive model loading across package versions
- **Real-time Web Scraping**: Robust Hacker News data collection with error handling
- **Production Ready**: Multiple deployment-ready Streamlit applications

### 📊 Performance Metrics
- **Model Accuracy**: 94.9% R² (Random Forest), 86.9% R² (Linear Regression)
- **Data Processing**: 77.6% success rate in salary parsing (742/956 records)
- **Environment Support**: Compatible with 3+ Python versions
- **User Experience**: < 2 second prediction response time

---

## 🚀 Quick Start Commands

```bash
# Complete setup in 3 commands
git clone <repository-url> && cd BackStage-October-2025
pip install -r requirements.txt && python regenerate_models.py
streamlit run fresh_app.py --server.port 8506
```

**🎉 Your AI Salary Predictor will be running at `http://localhost:8506`**

---

*Built with ❤️ using Python, Streamlit, scikit-learn, and modern ML practices*