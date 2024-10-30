import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

def prepare_data(df):
    """Prepare data for model training."""
    # Ensure required columns exist
    required_columns = ['ContractLength', 'MonthlyCharges', 'Churn']
    if not all(col in df.columns for col in required_columns):
        raise ValueError("Missing required columns in the dataset")
    
    # Convert Churn to binary
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Prepare features and target
    X = df[['ContractLength', 'MonthlyCharges']]
    y = df['Churn']
    
    return X, y

def train_model(X, y):
    """Train logistic regression model and return metrics."""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Get predictions
    y_pred = model.predict(X_test_scaled)
    probabilities = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    # Get feature importance
    feature_importance = abs(model.coef_[0])
    
    return model, metrics, probabilities, feature_importance
