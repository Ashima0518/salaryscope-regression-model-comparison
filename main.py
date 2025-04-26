#!/usr/bin/env python3
"""
SalaryScope: Predictive Modeling & Benchmarking with Regression

This script runs the complete pipeline for the SalaryScope project:
1. Load or generate synthetic salary data
2. Preprocess the data for machine learning
3. Train multiple regression models
4. Evaluate and compare model performance
5. Visualize results
"""

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import time

# Import project modules
from src.data_processing import generate_synthetic_data, save_data, load_data, preprocess_data
from src.model_training import train_all_models, save_models
from src.model_evaluation import evaluate_models, find_best_model, save_evaluation_results, generate_evaluation_report
from src.visualization import (
    plot_feature_distributions, plot_feature_importance, 
    plot_model_comparison, plot_prediction_vs_actual, plot_correlation_matrix
)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='SalaryScope: Salary Prediction Model Comparison')
    
    parser.add_argument('--samples', type=int, default=1000,
                        help='Number of samples to generate (default: 1000)')
    
    parser.add_argument('--regenerate', action='store_true',
                        help='Force regeneration of synthetic data even if it exists')
    
    parser.add_argument('--skip-plots', action='store_true',
                        help='Skip generating plots')
    
    parser.add_argument('--save-models', action='store_true',
                        help='Save trained models to disk')
    
    return parser.parse_args()

def main():
    """Main function to run the complete pipeline"""
    args = parse_arguments()
    
    print("\n" + "="*80)
    print(" "*30 + "SALARYSCOPE")
    print(" "*15 + "Predictive Modeling & Benchmarking with Regression")
    print("="*80 + "\n")
    
    # Create directories if they don't exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Step 1: Load or generate data
    data_path = 'data/salary_data.csv'
    
    if args.regenerate or not os.path.exists(data_path):
        print(f"\n[1/5] Generating synthetic salary data with {args.samples} samples...")
        data = generate_synthetic_data(n_samples=args.samples)
        save_data(data, filename='salary_data.csv')
    else:
        print("\n[1/5] Loading existing salary data...")
        data = load_data(data_path)
        
    print(f"Dataset shape: {data.shape}")
    print("Sample data:")
    print(data.head())
    
    # Step 2: Preprocess data
    print("\n[2/5] Preprocessing data...")
    X_train_processed, X_test_processed, y_train, y_test, preprocessor = preprocess_data(data)
    print(f"Training set shape: {X_train_processed.shape}")
    print(f"Test set shape: {X_test_processed.shape}")
    
    # Get feature names after preprocessing
    categorical_features = ['education', 'job_role', 'location', 'company_size']
    numeric_features = ['years_experience']
    
    # Get feature names from the preprocessor
    try:
        # Try to get feature names from OneHotEncoder
        cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_features = cat_encoder.get_feature_names_out(categorical_features)
        feature_names = np.concatenate([numeric_features, cat_features])
    except:
        # Fallback: just use feature indices
        feature_names = [f"feature_{i}" for i in range(X_train_processed.shape[1])]
    
    # Step 3: Train models
    print("\n[3/5] Training regression models...")
    start_time = time.time()
    models = train_all_models(X_train_processed, y_train)
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.2f} seconds")
    
    # Optionally save models
    if args.save_models:
        save_models(models)
    
    # Step 4: Evaluate models
    print("\n[4/5] Evaluating model performance...")
    results = evaluate_models(models, X_test_processed, y_test)
    print("\nModel Performance Metrics:")
    print(results)
    
    # Find best model for RMSE and R²
    best_rmse_model, best_rmse = find_best_model(results, 'RMSE', higher_is_better=False)
    best_r2_model, best_r2 = find_best_model(results, 'R²', higher_is_better=True)
    
    print(f"\nBest model by RMSE: {best_rmse_model} (RMSE = {best_rmse:.2f})")
    print(f"Best model by R²: {best_r2_model} (R² = {best_r2:.4f})")
    
    # Save evaluation results
    save_evaluation_results(results)
    
    # Generate evaluation report
    report = generate_evaluation_report(results)
    with open('results/evaluation_report.md', 'w') as f:
        f.write(report)
    print("Evaluation report saved to results/evaluation_report.md")
    
    # Step 5: Visualize results
    if not args.skip_plots:
        print("\n[5/5] Generating visualizations...")
        
        # Plot feature distributions
        plot_feature_distributions(data)
        
        # Plot correlation matrix
        plot_correlation_matrix(data)
        
        # Plot feature importance
        plot_feature_importance(models, feature_names)
        
        # Plot model comparison
        plot_model_comparison(results)
        
        # Plot predicted vs actual values
        plot_prediction_vs_actual(models, X_test_processed, y_test)
        
        print("Visualizations saved to plots/ directory")
    else:
        print("\n[5/5] Skipping visualizations (--skip-plots flag set)")
    
    print("\n" + "="*80)
    print(" "*25 + "PIPELINE COMPLETED SUCCESSFULLY")
    print("="*80 + "\n")

if __name__ == "__main__":
    main() 