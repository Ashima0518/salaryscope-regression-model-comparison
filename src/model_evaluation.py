import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import json
import os

def calculate_metrics(y_true, y_pred):
    """
    Calculate regression metrics.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
        
    Returns:
    --------
    dict
        Dictionary of metrics
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'RMSE': rmse,
        'R²': r2,
        'MAE': mae,
        'MAPE': mape
    }

def evaluate_model(model, X_test, y_test, model_name=None):
    """
    Evaluate a single model.
    
    Parameters:
    -----------
    model : object
        Trained model with predict method
    X_test : array-like
        Test features
    y_test : array-like
        True target values
    model_name : str, optional
        Name of the model
        
    Returns:
    --------
    dict
        Dictionary of metrics
    """
    y_pred = model.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred)
    
    if model_name:
        metrics['model'] = model_name
        
    return metrics

def evaluate_models(models, X_test, y_test):
    """
    Evaluate multiple models.
    
    Parameters:
    -----------
    models : dict
        Dictionary of trained models
    X_test : array-like
        Test features
    y_test : array-like
        True target values
        
    Returns:
    --------
    pd.DataFrame
        DataFrame of model metrics
    """
    results = []
    
    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test, name)
        results.append(metrics)
        
    return pd.DataFrame(results).set_index('model')

def find_best_model(evaluation_results, metric='RMSE', higher_is_better=False):
    """
    Find the best model based on a specific metric.
    
    Parameters:
    -----------
    evaluation_results : pd.DataFrame
        DataFrame of model metrics
    metric : str
        Metric to use for comparison
    higher_is_better : bool
        Whether higher values are better
        
    Returns:
    --------
    tuple
        (best_model_name, best_score)
    """
    if higher_is_better:
        best_idx = evaluation_results[metric].idxmax()
    else:
        best_idx = evaluation_results[metric].idxmin()
        
    best_score = evaluation_results.loc[best_idx, metric]
    
    return best_idx, best_score

def save_evaluation_results(results, filename='model_evaluation_results.json', directory='results'):
    """
    Save evaluation results to disk.
    
    Parameters:
    -----------
    results : pd.DataFrame
        DataFrame of model metrics
    filename : str
        Filename for saved results
    directory : str
        Directory to save results
        
    Returns:
    --------
    str
        Path to saved results
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    filepath = os.path.join(directory, filename)
    
    # Convert DataFrame to dictionary for JSON serialization
    results_dict = results.reset_index().to_dict(orient='records')
    
    with open(filepath, 'w') as f:
        json.dump(results_dict, f, indent=4)
        
    print(f"Saved evaluation results to {filepath}")
    return filepath

def generate_evaluation_report(results):
    """
    Generate a text report of model evaluation results.
    
    Parameters:
    -----------
    results : pd.DataFrame
        DataFrame of model metrics
        
    Returns:
    --------
    str
        Formatted report
    """
    report = "# Model Evaluation Report\n\n"
    report += "## Model Performance Metrics\n\n"
    
    # Format DataFrame as markdown table
    report += results.to_markdown() + "\n\n"
    
    # Find best models for each metric
    report += "## Best Models\n\n"
    
    for metric in ['RMSE', 'R²', 'MAE', 'MAPE']:
        higher_is_better = metric == 'R²'  # Only R² is better when higher
        best_model, best_score = find_best_model(results, metric, higher_is_better)
        
        report += f"- **{metric}**: {best_model} ({best_score:.4f})\n"
    
    return report 