#!/usr/bin/env python3
"""
Regenerate ML models in the main Python environment to ensure compatibility
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import joblib
import json
import os
import re

def parse_salary(salary_str):
    """Parse salary estimate string into numeric value"""
    if pd.isna(salary_str) or salary_str == '-1':
        return None
    try:
        # Extract numbers from salary string
        numbers = re.findall(r'\d+', str(salary_str))
        if len(numbers) >= 2:
            return (int(numbers[0]) + int(numbers[1])) * 500  # Simple conversion
        elif len(numbers) == 1:
            return int(numbers[0]) * 1000
        return None
    except:
        return None

def main():
    print("🔄 Regenerating models in main Python environment...")
    
    # Load the data
    df = pd.read_csv('glassdoor_jobs.csv')
    print(f"📊 Loaded data shape: {df.shape}")
    
    # Parse salaries
    df['parsed_salary'] = df['Salary Estimate'].apply(parse_salary)
    df_clean = df.dropna(subset=['parsed_salary']).copy()
    
    print(f"✅ Clean data shape: {df_clean.shape}")
    
    # Create features
    X = df_clean[['Rating', 'Founded']].copy()
    X['hourly'] = 0  # All salary positions
    X['employer_provided'] = 1  # All have salary info
    X['Job Title'] = df_clean['Job Title']
    X['Job Description'] = df_clean['Job Title']  # Simple description
    
    y = df_clean['parsed_salary']
    
    print(f"🎯 Target variable range: ${y.min():,.0f} - ${y.max():,.0f}")
    
    # Define preprocessing
    numeric_features = ['Rating', 'Founded', 'hourly', 'employer_provided']
    categorical_features = ['Job Title', 'Job Description']
    
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # Create pipelines
    rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    lr_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    
    # Train models
    print("🤖 Training models...")
    rf_pipeline.fit(X, y)
    lr_pipeline.fit(X, y)
    
    # Create clustering models
    X_processed = preprocessor.fit_transform(X)
    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(X_processed)
    
    pca = PCA(n_components=2, random_state=42)
    pca.fit(X_processed)
    
    print("✅ Models trained successfully!")
    print(f"📈 Random Forest R²: {rf_pipeline.score(X, y):.3f}")
    print(f"📈 Linear Regression R²: {lr_pipeline.score(X, y):.3f}")
    
    # Save models
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    
    print("💾 Saving models...")
    joblib.dump(rf_pipeline, os.path.join(models_dir, 'random_forest_model.pkl'))
    joblib.dump(lr_pipeline, os.path.join(models_dir, 'linear_regression_model.pkl'))
    joblib.dump(preprocessor, os.path.join(models_dir, 'preprocessor.pkl'))
    joblib.dump(kmeans, os.path.join(models_dir, 'kmeans_model.pkl'))
    joblib.dump(pca, os.path.join(models_dir, 'pca_model.pkl'))
    
    # Save metadata
    model_metadata = {
        'feature_columns': list(X.columns),
        'categorical_features': categorical_features,
        'numerical_features': numeric_features,
        'target_column': 'parsed_salary',
        'rf_r2': float(rf_pipeline.score(X, y)),
        'lr_r2': float(lr_pipeline.score(X, y)),
        'n_samples': len(X),
        'n_features': len(X.columns),
        'job_titles': sorted(df_clean['Job Title'].unique().tolist())
    }
    
    with open(os.path.join(models_dir, 'model_metadata.json'), 'w') as f:
        json.dump(model_metadata, f, indent=2)
    
    print(f"✅ Models and metadata saved to '{models_dir}' directory")
    print("📁 Available files:")
    for file in sorted(os.listdir(models_dir)):
        print(f"  - {file}")
    
    print("\n🎉 Model regeneration complete!")

if __name__ == "__main__":
    main()