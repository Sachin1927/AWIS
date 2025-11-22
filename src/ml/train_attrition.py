import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib
from pathlib import Path
import sys
import os

# Fix imports - add src directory to Python path
CURRENT_FILE = Path(__file__).resolve()
CURRENT_DIR = CURRENT_FILE.parent  # src/ml/
SRC_DIR = CURRENT_DIR.parent  # src/
PROJECT_ROOT = SRC_DIR.parent  # AWIS/

# Add src to path so we can import from utils, etc.
sys.path.insert(0, str(SRC_DIR))

# Now import from utils
from utils.logger import setup_logger
from ml.preprocessing import DataPreprocessor

logger = setup_logger(__name__)

class AttritionModelTrainer:
    """Train employee attrition prediction model"""
    
    def __init__(self, model_type: str = 'random_forest'):
        self.model_type = model_type
        self.model = None
        self.preprocessor = DataPreprocessor()
        self.feature_importance = None
    
    def load_data(self, data_path: str = None) -> pd.DataFrame:
        """Load employee data"""
        if data_path is None:
            data_path = PROJECT_ROOT / "data" / "employees.csv"
        
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        return df
    
    def train(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42
    ):
        """Train the model"""
        logger.info("Preparing data...")
        X, y = self.preprocessor.prepare_attrition_data(df)
        
        logger.info(f"Dataset shape: {X.shape}")
        logger.info(f"Target distribution:\n{y.value_counts()}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        
        # Scale features
        X_train_scaled = self.preprocessor.scale_features(
            X_train, scaler_name='attrition', fit=True
        )
        X_test_scaled = self.preprocessor.scale_features(
            X_test, scaler_name='attrition', fit=False
        )
        
        # Initialize model
        logger.info(f"Training {self.model_type} model...")
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=random_state,
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=random_state
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Train
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        logger.info("\n" + "="*60)
        logger.info("MODEL EVALUATION")
        logger.info("="*60)
        logger.info("\nClassification Report:")
        logger.info("\n" + classification_report(y_test, y_pred))
        logger.info(f"\nROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"\nConfusion Matrix:")
        logger.info(f"  TN: {cm[0,0]}, FP: {cm[0,1]}")
        logger.info(f"  FN: {cm[1,0]}, TP: {cm[1,1]}")
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("\nTop 10 Feature Importance:")
        logger.info(self.feature_importance.head(10).to_string(index=False))
        
        return self.model
    
    def save_model(self, output_dir: str = None):
        """Save trained model and scaler"""
        if output_dir is None:
            output_dir = PROJECT_ROOT / "4_models" / "attrition"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = output_dir / "model.pkl"
        joblib.dump(self.model, model_path)
        logger.info(f"\nModel saved to {model_path}")
        
        # Save scaler
        scaler_path = output_dir / "scaler.pkl"
        self.preprocessor.save_scaler('attrition', scaler_path)
        logger.info(f"Scaler saved to {scaler_path}")
        
        # Save feature importance
        feature_path = output_dir / "feature_importance.csv"
        self.feature_importance.to_csv(feature_path, index=False)
        logger.info(f"Feature importance saved to {feature_path}")


if __name__ == "__main__":
    logger.info("="*60)
    logger.info("ATTRITION MODEL TRAINING")
    logger.info("="*60)
    logger.info("")
    
    trainer = AttritionModelTrainer(model_type='random_forest')
    df = trainer.load_data()
    trainer.train(df)
    trainer.save_model()
    
    logger.info("")
    logger.info("="*60)
    logger.info("✅ ATTRITION MODEL TRAINING COMPLETE!")
    logger.info("="*60)