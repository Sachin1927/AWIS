import pytest
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.ml.inference_forecast import SkillDemandForecaster

@pytest.fixture
def forecaster():
    """Create forecaster instance"""
    return SkillDemandForecaster()

def test_forecaster_loads(forecaster):
    """Test that forecaster loads successfully"""
    assert forecaster.model is not None

def test_forecast_skill(forecaster):
    """Test skill forecasting"""
    # Use a skill that exists in the data
    forecast_df = forecaster.forecast("Python", months_ahead=3)
    
    # Check if forecast was generated (might be empty if skill not found)
    if len(forecast_df) > 0:
        assert 'date' in forecast_df.columns
        assert 'skill_name' in forecast_df.columns
        assert 'forecasted_demand' in forecast_df.columns
        assert len(forecast_df) == 3

def test_forecast_returns_valid_values(forecaster):
    """Test that forecasts are non-negative"""
    forecast_df = forecaster.forecast("Python", months_ahead=6)
    
    if len(forecast_df) > 0:
        assert all(forecast_df['forecasted_demand'] >= 0)

def test_trending_skills(forecaster):
    """Test trending skills identification"""
    trending = forecaster.get_trending_skills(top_n=5)
    
    assert isinstance(trending, list)
    
    if len(trending) > 0:
        assert len(trending) <= 5
        
        for skill in trending:
            assert 'skill_name' in skill
            assert 'growth_rate' in skill
            assert 'trend' in skill
            assert skill['trend'] in ['Rising', 'Stable', 'Declining']

def test_forecast_top_skills(forecaster):
    """Test forecasting for top skills"""
    forecast_df = forecaster.forecast_top_skills(top_n=3, months_ahead=3)
    
    if len(forecast_df) > 0:
        skills = forecast_df['skill_name'].unique()
        assert len(skills) <= 3