import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from pathlib import Path
import sys

# Fix imports
CURRENT_FILE = Path(__file__).resolve()
CURRENT_DIR = CURRENT_FILE.parent  # src/ml/
SRC_DIR = CURRENT_DIR.parent  # src/
PROJECT_ROOT = SRC_DIR.parent  # AWIS/

sys.path.insert(0, str(SRC_DIR))

from utils.logger import setup_logger
from ml.preprocessing import DataPreprocessor

logger = setup_logger(__name__)

class ForecastModelTrainer:
    """Train skill demand forecasting model"""
    
    def __init__(self):
        self.model = None
        self.preprocessor = DataPreprocessor()
    
    def load_data(self, data_path: str = None) -> pd.DataFrame:
        """Load historical skill demand data"""
        if data_path is None:
            data_path = PROJECT_ROOT / "data" / "historical_skill_demand.csv"
        
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        return df
    
    def train(self, df: pd.DataFrame, test_months: int = 6):
        """Train forecasting model"""
        logger.info("Preparing forecast data...")
        
        # Prepare data
        df = self.preprocessor.prepare_forecast_data(df)
        
        # Create lag features
        df = self.preprocessor.create_lag_features(df)
        
        logger.info(f"Data shape after preprocessing: {df.shape}")
        
        # Features and target
        feature_cols = [
            'year', 'month', 'quarter', 'days_since_start',
            'demand_count_lag_1', 'demand_count_lag_3',
            'demand_count_lag_6', 'demand_count_lag_12',
            'demand_count_rolling_mean_3', 'demand_count_rolling_std_3'
        ]
        
        X = df[feature_cols]
        y = df['demand_count']
        
        # Time-based split
        split_idx = len(df) - test_months
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        
        # Train model
        logger.info("Training Random Forest Regressor...")
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        logger.info("\n" + "="*60)
        logger.info("MODEL EVALUATION")
        logger.info("="*60)
        logger.info(f"\n  MAE: {mae:.2f}")
        logger.info(f"  RMSE: {rmse:.2f}")
        logger.info(f"  R²: {r2:.4f}")
        
        return self.model
    
    def save_model(self, output_dir: str = None):
        """Save trained model"""
        if output_dir is None:
            output_dir = PROJECT_ROOT / "4_models" / "forecast"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = output_dir / "model.pkl"
        joblib.dump(self.model, model_path)
        logger.info(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    logger.info("="*60)
    logger.info("FORECAST MODEL TRAINING")
    logger.info("="*60)
    logger.info("")
    
    trainer = ForecastModelTrainer()
    df = trainer.load_data()
    trainer.train(df)
    trainer.save_model()
    
    logger.info("")
    logger.info("="*60)
    logger.info("✅ FORECAST MODEL TRAINING COMPLETE!")
    logger.info("="*60)