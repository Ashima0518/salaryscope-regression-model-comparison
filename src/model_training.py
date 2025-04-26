import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import pickle
import os
from datetime import datetime

def train_linear_regression(X_train, y_train):
    """
    Train a Linear Regression model.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Target variable
        
    Returns:
    --------
    sklearn.linear_model.LinearRegression
        Trained Linear Regression model
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_decision_tree(X_train, y_train, max_depth=10, min_samples_split=5):
    """
    Train a Decision Tree Regression model.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Target variable
    max_depth : int
        Maximum depth of the tree
    min_samples_split : int
        Minimum samples required to split a node
        
    Returns:
    --------
    sklearn.tree.DecisionTreeRegressor
        Trained Decision Tree model
    """
    model = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train, n_estimators=100, max_depth=10):
    """
    Train a Random Forest Regression model.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Target variable
    n_estimators : int
        Number of trees in the forest
    max_depth : int
        Maximum depth of the trees
        
    Returns:
    --------
    sklearn.ensemble.RandomForestRegressor
        Trained Random Forest model
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1  # Use all available cores
    )
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train, n_estimators=100, learning_rate=0.1, max_depth=6):
    """
    Train an XGBoost Regression model.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Target variable
    n_estimators : int
        Number of boosting rounds
    learning_rate : float
        Step size shrinkage used to prevent overfitting
    max_depth : int
        Maximum depth of a tree
        
    Returns:
    --------
    xgboost.XGBRegressor
        Trained XGBoost model
    """
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def train_all_models(X_train, y_train):
    """
    Train all regression models.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Target variable
        
    Returns:
    --------
    dict
        Dictionary of trained models
    """
    models = {
        'Linear Regression': train_linear_regression(X_train, y_train),
        'Decision Tree': train_decision_tree(X_train, y_train),
        'Random Forest': train_random_forest(X_train, y_train),
        'XGBoost': train_xgboost(X_train, y_train)
    }
    return models

def save_models(models, directory='models'):
    """
    Save trained models to disk.
    
    Parameters:
    -----------
    models : dict
        Dictionary of trained models
    directory : str
        Directory to save models
        
    Returns:
    --------
    dict
        Dictionary of file paths for saved models
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepaths = {}
    
    for name, model in models.items():
        filename = f"{name.lower().replace(' ', '_')}_{timestamp}.pkl"
        filepath = os.path.join(directory, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
            
        filepaths[name] = filepath
        print(f"Saved {name} model to {filepath}")
        
    return filepaths

def load_model(filepath):
    """
    Load a trained model from disk.
    
    Parameters:
    -----------
    filepath : str
        Path to the saved model
        
    Returns:
    --------
    object
        Loaded model
    """
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    return model 