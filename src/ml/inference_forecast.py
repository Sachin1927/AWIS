import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to Python path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.append(str(project_root))

# Try to import project utilities; if not available (e.g. running outside package),
# provide minimal fallbacks so the module can still run.
try:
    # Try relative import if running as part of package
    from src.utils.logger import setup_logger
    from src.utils.config import PROJECT_ROOT
except ImportError:
    import logging

    def setup_logger(name):
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    PROJECT_ROOT = project_root

logger = setup_logger(__name__)

class SkillDemandForecaster:
    """Forecast future skill demand"""
    
    def __init__(self, model_dir: str = None):
        if model_dir is None:
            model_dir = PROJECT_ROOT / "4_models" / "forecast"
        
        self.model_dir = Path(model_dir)
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load trained forecasting model"""
        try:
            model_path = self.model_dir / "model.pkl"
            self.model = joblib.load(model_path)
            logger.info("Forecast model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def load_historical_data(self, skill_name: str = None) -> pd.DataFrame:
        """Load historical demand data"""
        data_path = PROJECT_ROOT / "data" / "historical_skill_demand.csv"
        df = pd.read_csv(data_path)
        df['date'] = pd.to_datetime(df['date'])
        
        if skill_name:
            df = df[df['skill_name'] == skill_name]
        
        return df.sort_values('date')
    
    def prepare_forecast_features(
        self,
        historical_data: pd.DataFrame,
        forecast_date: datetime
    ) -> np.ndarray:
        """Prepare features for forecasting"""
        # Get recent values for lag features
        recent_data = historical_data.tail(12).copy()
        
        # Extract date features
        year = forecast_date.year
        month = forecast_date.month
        quarter = (month - 1) // 3 + 1
        
        # Calculate days since start
        start_date = historical_data['date'].min()
        days_since_start = (forecast_date - start_date).days
        
        # Get lag values
        lag_1 = recent_data['demand_count'].iloc[-1] if len(recent_data) >= 1 else 0
        lag_3 = recent_data['demand_count'].iloc[-3] if len(recent_data) >= 3 else lag_1
        lag_6 = recent_data['demand_count'].iloc[-6] if len(recent_data) >= 6 else lag_1
        lag_12 = recent_data['demand_count'].iloc[-12] if len(recent_data) >= 12 else lag_1
        
        # Rolling statistics
        rolling_mean = recent_data['demand_count'].tail(3).mean()
        rolling_std = recent_data['demand_count'].tail(3).std()
        
        features = np.array([
            year, month, quarter, days_since_start,
            lag_1, lag_3, lag_6, lag_12,
            rolling_mean, rolling_std
        ]).reshape(1, -1)
        
        return features
    
    def forecast(
        self,
        skill_name: str,
        months_ahead: int = 6
    ) -> pd.DataFrame:
        """
        Forecast skill demand for future months
        
        Args:
            skill_name: Name of skill to forecast
            months_ahead: Number of months to forecast
        
        Returns:
            DataFrame with forecasted demand
        """
        # Load historical data
        historical_data = self.load_historical_data(skill_name)
        
        if len(historical_data) == 0:
            logger.warning(f"No historical data for skill: {skill_name}")
            return pd.DataFrame()
        
        # Get last date
        last_date = historical_data['date'].max()
        
        # Generate forecasts
        forecasts = []
        
        for i in range(1, months_ahead + 1):
            forecast_date = last_date + timedelta(days=30 * i)
            
            # Prepare features
            X = self.prepare_forecast_features(historical_data, forecast_date)
            
            # Predict
            demand = self.model.predict(X)[0]
            demand = max(0, int(demand))  # Ensure non-negative
            
            forecasts.append({
                'date': forecast_date.strftime('%Y-%m-%d'),
                'skill_name': skill_name,
                'forecasted_demand': demand,
                'confidence': 'Medium'  # Could be improved with prediction intervals
            })
            
            # Update historical data with forecast for next iteration
            new_row = pd.DataFrame({
                'date': [forecast_date],
                'skill_name': [skill_name],
                'demand_count': [demand]
            })
            historical_data = pd.concat([historical_data, new_row], ignore_index=True)
        
        return pd.DataFrame(forecasts)
    
    def forecast_top_skills(
        self,
        top_n: int = 10,
        months_ahead: int = 6
    ) -> pd.DataFrame:
        """Forecast demand for top N skills"""
        # Get all skills
        historical_data = self.load_historical_data()
        
        # Find top skills by recent demand
        recent_demand = historical_data.groupby('skill_name')['demand_count'].mean()
        top_skills = recent_demand.nlargest(top_n).index.tolist()
        
        # Forecast each
        all_forecasts = []
        
        for skill in top_skills:
            try:
                forecast = self.forecast(skill, months_ahead)
                all_forecasts.append(forecast)
            except Exception as e:
                logger.error(f"Error forecasting {skill}: {e}")
        
        if all_forecasts:
            return pd.concat(all_forecasts, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def get_trending_skills(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """Identify trending skills based on growth rate"""
        historical_data = self.load_historical_data()
        
        trends = []
        
        for skill in historical_data['skill_name'].unique():
            skill_data = historical_data[
                historical_data['skill_name'] == skill
            ].sort_values('date')
            
            if len(skill_data) >= 12:
                # Compare last 3 months to previous 3 months
                recent = skill_data.tail(3)['demand_count'].mean()
                previous = skill_data.iloc[-6:-3]['demand_count'].mean()
                
                if previous > 0:
                    growth_rate = ((recent - previous) / previous) * 100
                else:
                    growth_rate = 0
                
                trends.append({
                    'skill_name': skill,
                    'growth_rate': growth_rate,
                    'current_demand': int(recent),
                    'trend': 'Rising' if growth_rate > 5 else 'Stable' if growth_rate > -5 else 'Declining'
                })
        
        # Sort by growth rate
        trends.sort(key=lambda x: x['growth_rate'], reverse=True)
        
        return trends[:top_n]

# Singleton instance
_forecaster_instance = None

def get_forecaster() -> SkillDemandForecaster:
    """Get or create singleton forecaster instance"""
    global _forecaster_instance
    if _forecaster_instance is None:
        _forecaster_instance = SkillDemandForecaster()
    return _forecaster_instance