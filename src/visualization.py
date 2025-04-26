import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

def set_plotting_style():
    """Set the plotting style for visualizations"""
    sns.set(style="whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    
def plot_feature_distributions(data, save_dir='plots'):
    """
    Plot the distributions of features in the dataset.
    
    Parameters:
    -----------
    data : pd.DataFrame
        The dataset
    save_dir : str
        Directory to save plots
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    set_plotting_style()
    
    # Numerical features
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data['years_experience'], kde=True, ax=ax)
    ax.set_title('Distribution of Years of Experience')
    ax.set_xlabel('Years of Experience')
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'years_experience_distribution.png'))
    plt.close(fig)
    
    # Categorical features
    categorical_cols = ['education', 'job_role', 'location', 'company_size']
    
    for col in categorical_cols:
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.countplot(y=col, data=data, order=data[col].value_counts().index, ax=ax)
        ax.set_title(f'Distribution of {col.replace("_", " ").title()}')
        ax.set_xlabel('Count')
        ax.set_ylabel(col.replace('_', ' ').title())
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f'{col}_distribution.png'))
        plt.close(fig)
    
    # Target variable
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data['salary'], kde=True, ax=ax)
    ax.set_title('Distribution of Salaries')
    ax.set_xlabel('Salary ($)')
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'salary_distribution.png'))
    plt.close(fig)
    
def plot_feature_importance(models, feature_names, save_dir='plots'):
    """
    Plot feature importance for tree-based models.
    
    Parameters:
    -----------
    models : dict
        Dictionary of trained models
    feature_names : list
        List of feature names
    save_dir : str
        Directory to save plots
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    set_plotting_style()
    
    tree_models = {
        name: model for name, model in models.items() 
        if name in ['Decision Tree', 'Random Forest', 'XGBoost']
    }
    
    for name, model in tree_models.items():
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            # Create a DataFrame for easier plotting
            if isinstance(feature_names, np.ndarray) and feature_names.ndim > 1:
                # Handle one-hot encoded features
                importance_df = pd.DataFrame({
                    'Feature': [f"Feature {i}" for i in range(len(importances))],
                    'Importance': importances
                })
            else:
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                })
            
            # Sort by importance
            importance_df = importance_df.sort_values('Importance', ascending=False)
            
            # Plot top 15 features or all if less than 15
            n_features = min(15, len(importance_df))
            top_features = importance_df.head(n_features)
            
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.barplot(x='Importance', y='Feature', data=top_features, ax=ax)
            ax.set_title(f'Feature Importance - {name}')
            fig.tight_layout()
            fig.savefig(os.path.join(save_dir, f'{name.lower().replace(" ", "_")}_feature_importance.png'))
            plt.close(fig)
    
def plot_model_comparison(results, save_dir='plots'):
    """
    Plot model comparison based on evaluation metrics.
    
    Parameters:
    -----------
    results : pd.DataFrame
        DataFrame of model metrics
    save_dir : str
        Directory to save plots
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    set_plotting_style()
    
    # Reshape results for easier plotting
    results_melted = results.reset_index().melt(
        id_vars='model', 
        var_name='Metric', 
        value_name='Value'
    )
    
    # Plot for RMSE and MAE (lower is better)
    error_metrics = results_melted[results_melted['Metric'].isin(['RMSE', 'MAE'])]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.barplot(x='model', y='Value', hue='Metric', data=error_metrics, ax=ax)
    ax.set_title('Model Comparison - Error Metrics (Lower is Better)')
    ax.set_xlabel('Model')
    ax.set_ylabel('Value')
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'model_comparison_error_metrics.png'))
    plt.close(fig)
    
    # Plot for R² (higher is better)
    r2_metrics = results_melted[results_melted['Metric'] == 'R²']
    
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.barplot(x='model', y='Value', data=r2_metrics, ax=ax, palette='viridis')
    ax.set_title('Model Comparison - R² (Higher is Better)')
    ax.set_xlabel('Model')
    ax.set_ylabel('R² Value')
    ax.set_ylim(0, 1)  # R² typically ranges from 0 to 1
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'model_comparison_r2.png'))
    plt.close(fig)
    
def plot_prediction_vs_actual(models, X_test, y_test, save_dir='plots'):
    """
    Plot predicted vs actual values for each model.
    
    Parameters:
    -----------
    models : dict
        Dictionary of trained models
    X_test : array-like
        Test features
    y_test : array-like
        True target values
    save_dir : str
        Directory to save plots
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    set_plotting_style()
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot perfect prediction line
        max_val = max(np.max(y_test), np.max(y_pred))
        min_val = min(np.min(y_test), np.min(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        
        # Plot scatter of predictions
        ax.scatter(y_test, y_pred, alpha=0.5)
        
        ax.set_title(f'Predicted vs Actual - {name}')
        ax.set_xlabel('Actual Salary ($)')
        ax.set_ylabel('Predicted Salary ($)')
        ax.legend()
        
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f'{name.lower().replace(" ", "_")}_pred_vs_actual.png'))
        plt.close(fig)
        
    # Create a combined plot for all models
    fig, axs = plt.subplots(2, 2, figsize=(18, 14))
    axs = axs.flatten()
    
    for i, (name, model) in enumerate(models.items()):
        y_pred = model.predict(X_test)
        
        ax = axs[i]
        
        # Plot perfect prediction line
        max_val = max(np.max(y_test), np.max(y_pred))
        min_val = min(np.min(y_test), np.min(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        
        # Plot scatter of predictions
        ax.scatter(y_test, y_pred, alpha=0.5)
        
        ax.set_title(f'{name}')
        ax.set_xlabel('Actual Salary ($)')
        ax.set_ylabel('Predicted Salary ($)')
        ax.legend()
    
    fig.suptitle('Predicted vs Actual Salaries - All Models', fontsize=16)
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)
    fig.savefig(os.path.join(save_dir, 'all_models_pred_vs_actual.png'))
    plt.close(fig)
    
def plot_correlation_matrix(data, save_dir='plots'):
    """
    Plot correlation matrix for numerical features and target.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Dataset
    save_dir : str
        Directory to save plots
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    set_plotting_style()
    
    # Select numerical columns
    numerical_data = data.select_dtypes(include=[np.number])
    
    # Calculate correlation matrix
    corr_matrix = numerical_data.corr()
    
    # Plot correlation matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Matrix')
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'correlation_matrix.png'))
    plt.close(fig) 