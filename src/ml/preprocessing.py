import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, List, Dict, Any
import joblib

class DataPreprocessor:
    """Preprocessing utilities for ML models"""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
    
    def prepare_attrition_data(
        self,
        df: pd.DataFrame,
        target_col: str = 'attrition'
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for attrition model"""
        # Select features
        feature_cols = [
            'age',
            'years_at_company',
            'monthly_income',
            'performance_rating',
            'satisfaction_level',
            'last_promotion_years',
            'training_hours'
        ]
        
        # Add department encoding if available
        if 'department' in df.columns:
            df = df.copy()
            df['department_encoded'] = LabelEncoder().fit_transform(df['department'])
            feature_cols.append('department_encoded')
        
        X = df[feature_cols].copy()
        y = df[target_col] if target_col in df.columns else None
        
        # Handle missing values
        X = X.fillna(X.median())
        
        return X, y
    
    def prepare_forecast_data(
        self,
        df: pd.DataFrame,
        skill_name: str = None
    ) -> pd.DataFrame:
        """Prepare time series data for forecasting"""
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        # Filter by skill if specified
        if skill_name:
            df = df[df['skill_name'] == skill_name]
        
        # Sort by date
        df = df.sort_values('date')
        
        # Create time-based features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['days_since_start'] = (df['date'] - df['date'].min()).dt.days
        
        return df
    
    def create_lag_features(
        self,
        df: pd.DataFrame,
        target_col: str = 'demand_count',
        lags: List[int] = [1, 3, 6, 12]
    ) -> pd.DataFrame:
        """Create lag features for time series"""
        df = df.copy()
        
        for lag in lags:
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        
        # Rolling statistics
        df[f'{target_col}_rolling_mean_3'] = df[target_col].rolling(3).mean()
        df[f'{target_col}_rolling_std_3'] = df[target_col].rolling(3).std()
        
        # Drop NaN rows created by lagging
        df = df.dropna()
        
        return df
    
    def scale_features(
        self,
        X: pd.DataFrame,
        scaler_name: str = 'default',
        fit: bool = True
    ) -> np.ndarray:
        """Scale features using StandardScaler"""
        if fit or scaler_name not in self.scalers:
            self.scalers[scaler_name] = StandardScaler()
            return self.scalers[scaler_name].fit_transform(X)
        else:
            return self.scalers[scaler_name].transform(X)
    
    def save_scaler(self, scaler_name: str, path: str):
        """Save scaler to disk"""
        if scaler_name in self.scalers:
            joblib.dump(self.scalers[scaler_name], path)
    
    def load_scaler(self, scaler_name: str, path: str):
        """Load scaler from disk"""
        self.scalers[scaler_name] = joblib.load(path)