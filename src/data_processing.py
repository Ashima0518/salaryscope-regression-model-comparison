import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

def generate_synthetic_data(n_samples=1000, random_state=42):
    """
    Generate synthetic salary data based on various features.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing synthetic salary data
    """
    np.random.seed(random_state)
    
    # Generate features
    years_experience = np.random.normal(5, 3, n_samples).round(1)
    years_experience = np.maximum(0, years_experience)  # No negative experience
    
    education_levels = ['High School', 'Bachelor', 'Master', 'PhD']
    education = np.random.choice(education_levels, n_samples, p=[0.2, 0.5, 0.2, 0.1])
    
    job_roles = ['Data Analyst', 'Software Engineer', 'Project Manager', 
                'Marketing Specialist', 'Sales Representative', 'HR Manager']
    job_role = np.random.choice(job_roles, n_samples)
    
    locations = ['New York', 'San Francisco', 'Chicago', 'Austin', 'Seattle', 'Remote']
    location = np.random.choice(locations, n_samples)
    
    company_size = np.random.choice(['Small', 'Medium', 'Large'], n_samples)
    
    # Some domain knowledge for base salary ranges
    base_salaries = {
        'Data Analyst': 70000,
        'Software Engineer': 90000,
        'Project Manager': 85000,
        'Marketing Specialist': 65000,
        'Sales Representative': 60000,
        'HR Manager': 75000
    }
    
    education_multipliers = {
        'High School': 0.8,
        'Bachelor': 1.0,
        'Master': 1.2,
        'PhD': 1.4
    }
    
    location_multipliers = {
        'New York': 1.3,
        'San Francisco': 1.4,
        'Chicago': 1.1, 
        'Austin': 1.05,
        'Seattle': 1.2,
        'Remote': 0.95
    }
    
    company_size_multipliers = {
        'Small': 0.9,
        'Medium': 1.0,
        'Large': 1.15
    }
    
    # Calculate base salaries
    base_salary = np.array([base_salaries[role] for role in job_role])
    
    # Apply multipliers
    education_mult = np.array([education_multipliers[level] for level in education])
    location_mult = np.array([location_multipliers[loc] for loc in location])
    company_mult = np.array([company_size_multipliers[size] for size in company_size])
    
    # Experience factor (diminishing returns after 10 years)
    experience_factor = 1 + 0.05 * np.minimum(years_experience, 10) + 0.01 * np.maximum(0, years_experience - 10)
    
    # Calculate salary with some noise
    salary = base_salary * education_mult * location_mult * company_mult * experience_factor
    noise = np.random.normal(0, 0.1, n_samples)  # 10% noise
    salary = salary * (1 + noise)
    
    # Create DataFrame
    data = pd.DataFrame({
        'years_experience': years_experience,
        'education': education,
        'job_role': job_role,
        'location': location,
        'company_size': company_size,
        'salary': salary.round(2)
    })
    
    return data

def save_data(data, filename='salary_data.csv', directory='data'):
    """Save the generated data to a CSV file"""
    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, filename)
    data.to_csv(filepath, index=False)
    print(f"Data saved to {filepath}")
    return filepath

def load_data(filepath='data/salary_data.csv'):
    """Load data from a CSV file"""
    return pd.read_csv(filepath)

def preprocess_data(data):
    """
    Preprocess the data for machine learning models.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input data
        
    Returns:
    --------
    tuple
        X_train, X_test, y_train, y_test, preprocessor
    """
    # Split features and target
    X = data.drop('salary', axis=1)
    y = data['salary']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Identify numeric and categorical features
    numeric_features = ['years_experience']
    categorical_features = ['education', 'job_role', 'location', 'company_size']
    
    # Create preprocessor
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # Fit the preprocessor
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    return X_train_processed, X_test_processed, y_train, y_test, preprocessor

if __name__ == "__main__":
    # Generate and save synthetic data
    data = generate_synthetic_data(n_samples=1000)
    filepath = save_data(data)
    print(f"Generated {len(data)} samples of synthetic salary data")
    print(data.head())
    print("\nData statistics:")
    print(data.describe()) 