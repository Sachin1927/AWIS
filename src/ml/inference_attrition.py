import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, List, Any
import sys

CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parent.parent
PROJECT_ROOT = SRC_DIR.parent

sys.path.insert(0, str(SRC_DIR))

from utils.logger import setup_logger

logger = setup_logger(__name__)

class AttritionPredictor:
    """Make attrition predictions for employees"""
    
    def __init__(self, model_dir=None):
        if model_dir is None:
            model_dir = PROJECT_ROOT / "4_models" / "attrition"
        
        self.model_dir = Path(model_dir)
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.load_model()
    
    def load_model(self):
        """Load trained model and scaler"""
        try:
            model_path = self.model_dir / "model.pkl"
            scaler_path = self.model_dir / "scaler.pkl"
            
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            
            if hasattr(self.scaler, 'feature_names_in_'):
                self.feature_names = list(self.scaler.feature_names_in_)
            else:
                self.feature_names = [
                    'age', 'years_at_company', 'monthly_income',
                    'performance_rating', 'satisfaction_level',
                    'last_promotion_years', 'training_hours', 'department_encoded'
                ]
            
            logger.info(f"Attrition model loaded ({len(self.feature_names)} features)")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def prepare_features(self, employee_data):
        """Prepare features as DataFrame"""
        features = {
            'age': employee_data.get('age', 30),
            'years_at_company': employee_data.get('years_at_company', 5),
            'monthly_income': employee_data.get('monthly_income', 50000),
            'performance_rating': employee_data.get('performance_rating', 3),
            'satisfaction_level': employee_data.get('satisfaction_level', 0.7),
            'last_promotion_years': employee_data.get('last_promotion_years', 2),
            'training_hours': employee_data.get('training_hours', 20),
            'department_encoded': employee_data.get('department_encoded', 0)
        }
        
        df = pd.DataFrame([features], columns=self.feature_names)
        return df
    
    def predict(self, employee_data):
        """Predict attrition for a single employee"""
        X = self.prepare_features(employee_data)
        X_scaled = self.scaler.transform(X)
        
        prediction = self.model.predict(X_scaled)[0]
        probability = self.model.predict_proba(X_scaled)[0][1]
        
        if probability >= 0.7:
            risk_level = "High"
        elif probability >= 0.4:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        return {
            'employee_id': employee_data.get('employee_id', 'Unknown'),
            'attrition_prediction': int(prediction),
            'attrition_probability': float(probability),
            'risk_level': risk_level,
            'will_leave': bool(prediction == 1)
        }
    
    def predict_batch(self, employees_df):
        """Predict attrition for multiple employees"""
        results = []
        for idx, employee in employees_df.iterrows():
            employee_dict = employee.to_dict()
            prediction = self.predict(employee_dict)
            results.append(prediction)
        return pd.DataFrame(results)
    
    def get_retention_recommendations(self, employee_data, prediction):
        """Generate retention recommendations"""
        recommendations = []
        
        if prediction['risk_level'] == 'High':
            if employee_data.get('satisfaction_level', 1.0) < 0.5:
                recommendations.append("Critical: Schedule 1-on-1 meeting")
                recommendations.append("Consider role adjustment")
            
            if employee_data.get('last_promotion_years', 0) > 3:
                recommendations.append("Evaluate for promotion or raise")
            
            if employee_data.get('training_hours', 0) < 20:
                recommendations.append("Provide development opportunities")
        elif prediction['risk_level'] == 'Medium':
            recommendations.append("Monitor engagement levels")
            recommendations.append("Schedule regular check-ins")
        else:
            recommendations.append("Employee appears engaged")
        
        return recommendations


_predictor_instance = None

def get_predictor():
    """Get or create singleton predictor instance"""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = AttritionPredictor()
    return _predictor_instance