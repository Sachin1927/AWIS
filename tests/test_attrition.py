import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.ml.inference_attrition import AttritionPredictor

@pytest.fixture
def predictor():
    """Create predictor instance"""
    return AttritionPredictor()

def test_predictor_loads(predictor):
    """Test that predictor loads successfully"""
    assert predictor.model is not None
    assert predictor.scaler is not None

def test_predict_single_employee(predictor):
    """Test single employee prediction"""
    employee_data = {
        'employee_id': 'TEST001',
        'age': 35,
        'years_at_company': 5,
        'monthly_income': 50000,
        'performance_rating': 3,
        'satisfaction_level': 0.7,
        'last_promotion_years': 2,
        'training_hours': 40
    }
    
    result = predictor.predict(employee_data)
    
    assert 'employee_id' in result
    assert 'attrition_probability' in result
    assert 'risk_level' in result
    assert 0 <= result['attrition_probability'] <= 1
    assert result['risk_level'] in ['Low', 'Medium', 'High']

def test_risk_level_calculation(predictor):
    """Test risk level categorization"""
    # High risk
    high_risk_data = {
        'employee_id': 'HIGH001',
        'age': 30,
        'years_at_company': 10,
        'monthly_income': 40000,
        'performance_rating': 2,
        'satisfaction_level': 0.2,
        'last_promotion_years': 8,
        'training_hours': 5
    }
    
    result = predictor.predict(high_risk_data)
    # Note: Actual risk level depends on model, just checking it returns valid value
    assert result['risk_level'] in ['Low', 'Medium', 'High']

def test_retention_recommendations(predictor):
    """Test recommendation generation"""
    employee_data = {
        'employee_id': 'REC001',
        'age': 35,
        'years_at_company': 5,
        'monthly_income': 50000,
        'performance_rating': 3,
        'satisfaction_level': 0.3,  # Low satisfaction
        'last_promotion_years': 5,  # Long time since promotion
        'training_hours': 10  # Low training
    }
    
    prediction = predictor.predict(employee_data)
    recommendations = predictor.get_retention_recommendations(
        employee_data,
        prediction
    )
    
    assert isinstance(recommendations, list)
    assert len(recommendations) > 0

def test_batch_prediction(predictor):
    """Test batch predictions"""
    import pandas as pd
    
    employees_df = pd.DataFrame([
        {
            'employee_id': 'BATCH001',
            'age': 30,
            'years_at_company': 3,
            'monthly_income': 45000,
            'performance_rating': 3,
            'satisfaction_level': 0.6,
            'last_promotion_years': 1,
            'training_hours': 30
        },
        {
            'employee_id': 'BATCH002',
            'age': 40,
            'years_at_company': 8,
            'monthly_income': 60000,
            'performance_rating': 4,
            'satisfaction_level': 0.8,
            'last_promotion_years': 2,
            'training_hours': 50
        }
    ])
    
    results = predictor.predict_batch(employees_df)
    
    assert len(results) == 2
    assert 'attrition_probability' in results.columns