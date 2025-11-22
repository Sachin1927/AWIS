"""
Load all ML models for AWIS
This script ensures all models are trained and ready to use
"""

import os
import sys
from pathlib import Path

# Fix imports - add project root to path
CURRENT_DIR = Path(__file__).parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Now we can import (change 'src' to '2_src')
from utils.logger import setup_logger

logger = setup_logger(__name__)


class ModelLoader:
    """Load and validate all ML models"""
    
    def __init__(self):
        self.models_dir = PROJECT_ROOT / "4_models"
        self.data_dir = PROJECT_ROOT / "data"
        
        # Model paths
        self.attrition_model_path = self.models_dir / "attrition" / "model.pkl"
        self.forecast_model_path = self.models_dir / "forecast" / "model.pkl"
        self.mobility_model_path = self.models_dir / "mobility" / "node2vec.model"
        
        # Predictor instances
        self.attrition_predictor = None
        self.forecaster = None
        self.mobility_analyzer = None
    
    def check_data_exists(self):
        """Check if synthetic data exists"""
        required_files = [
            "employees.csv",
            "skills.csv",
            "employee_skills.csv",
            "historical_skill_demand.csv"
        ]
        
        missing = []
        for file in required_files:
            filepath = self.data_dir / file
            if not filepath.exists():
                missing.append(file)
        
        if missing:
            logger.warning(f"Missing data files: {', '.join(missing)}")
            return False
        
        logger.info("✅ All data files exist")
        return True
    
    def check_models_exist(self):
        """Check which models exist"""
        models_status = {
            'attrition': self.attrition_model_path.exists(),
            'forecast': self.forecast_model_path.exists(),
            'mobility': self.mobility_model_path.exists()
        }
        
        for model_name, exists in models_status.items():
            if exists:
                logger.info(f"✅ {model_name.capitalize()} model exists")
            else:
                logger.warning(f"❌ {model_name.capitalize()} model missing")
        
        return models_status
    
    def generate_data(self):
        """Generate synthetic data"""
        logger.info("=" * 60)
        logger.info("Generating synthetic data...")
        logger.info("=" * 60)
        
        # Import here to avoid circular imports
        sys.path.insert(0, str(PROJECT_ROOT / "2_src"))
        from data_synth.generate_data import SyntheticDataGenerator
        
        generator = SyntheticDataGenerator()
        generator.save_all()
        
        logger.info("✅ Data generation complete")
    
    def train_attrition_model(self):
        """Train attrition prediction model"""
        logger.info("=" * 60)
        logger.info("Training attrition model...")
        logger.info("=" * 60)
        
        # Import here
        sys.path.insert(0, str(PROJECT_ROOT / "2_src"))
        from ml.train_attrition import AttritionModelTrainer
        
        trainer = AttritionModelTrainer(model_type='random_forest')
        df = trainer.load_data()
        trainer.train(df)
        trainer.save_model()
        
        logger.info("✅ Attrition model trained and saved")
    
    def train_forecast_model(self):
        """Train skill demand forecasting model"""
        logger.info("=" * 60)
        logger.info("Training forecast model...")
        logger.info("=" * 60)
        
        # Import here
        sys.path.insert(0, str(PROJECT_ROOT / "2_src"))
        from ml.train_forecast import ForecastModelTrainer
        
        trainer = ForecastModelTrainer()
        df = trainer.load_data()
        trainer.train(df)
        trainer.save_model()
        
        logger.info("✅ Forecast model trained and saved")
    
    def train_mobility_model(self):
        """Train mobility graph model"""
        logger.info("=" * 60)
        logger.info("Training mobility model...")
        logger.info("=" * 60)
        
        # Import here
        sys.path.insert(0, str(PROJECT_ROOT / "2_src"))
        from ml.train_mobility import MobilityGraphTrainer
        
        trainer = MobilityGraphTrainer()
        employees, employee_skills = trainer.load_data()
        trainer.build_graph(employees, employee_skills)
        trainer.train_node2vec()
        trainer.save_model()
        
        logger.info("✅ Mobility model trained and saved")
    
    def load_all_models(self):
        """Load all trained models into memory"""
        logger.info("=" * 60)
        logger.info("Loading all models into memory...")
        logger.info("=" * 60)
        
        try:
            sys.path.insert(0, str(PROJECT_ROOT / "2_src"))
            
            from ml.inference_attrition import get_predictor
            from ml.inference_forecast import get_forecaster
            from ml.inference_mobility import get_analyzer
            
            self.attrition_predictor = get_predictor()
            logger.info("✅ Attrition predictor loaded")
            
            self.forecaster = get_forecaster()
            logger.info("✅ Forecaster loaded")
            
            self.mobility_analyzer = get_analyzer()
            logger.info("✅ Mobility analyzer loaded")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def setup_all(self, force_retrain=False):
        """
        Complete setup: data generation + model training
        
        Args:
            force_retrain: If True, retrain models even if they exist
        """
        logger.info("\n")
        logger.info("=" * 60)
        logger.info("       AWIS MODEL SETUP")
        logger.info("=" * 60)
        logger.info("\n")
        
        # Step 1: Check/Generate Data
        logger.info("📊 Step 1: Checking data...")
        if not self.check_data_exists():
            logger.info("🔄 Generating synthetic data...")
            self.generate_data()
        else:
            logger.info("✅ Data already exists, skipping generation")
        
        logger.info("\n")
        
        # Step 2: Check/Train Models
        logger.info("🤖 Step 2: Checking models...")
        models_status = self.check_models_exist()
        
        logger.info("\n")
        
        # Train missing or force retrain
        if not models_status['attrition'] or force_retrain:
            self.train_attrition_model()
            logger.info("\n")
        
        if not models_status['forecast'] or force_retrain:
            self.train_forecast_model()
            logger.info("\n")
        
        if not models_status['mobility'] or force_retrain:
            self.train_mobility_model()
            logger.info("\n")
        
        # Step 3: Load Models
        logger.info("📥 Step 3: Loading models into memory...")
        success = self.load_all_models()
        
        logger.info("\n")
        logger.info("=" * 60)
        
        if success:
            logger.info("✅ SUCCESS! ALL MODELS READY!")
            logger.info("=" * 60)
            logger.info("\nNext steps:")
            logger.info("  1. Build RAG index:")
            logger.info("     python 2_src/rag_index/build_index.py")
            logger.info("\n  2. Install Ollama:")
            logger.info("     curl -fsSL https://ollama.com/install.sh | sh")
            logger.info("     ollama pull llama2")
            logger.info("\n  3. Run API:")
            logger.info("     cd 2_src/api && python main.py")
            logger.info("\n  4. Run Frontend:")
            logger.info("     streamlit run 3_app/streamlit_app.py")
        else:
            logger.error("❌ SETUP FAILED - Check errors above")
            logger.info("=" * 60)
        
        logger.info("\n")
        return success
    
    def validate_models(self):
        """Test that all models work correctly"""
        logger.info("\n")
        logger.info("=" * 60)
        logger.info("🧪 Validating models...")
        logger.info("=" * 60)
        logger.info("\n")
        
        all_passed = True
        
        try:
            # Test attrition
            logger.info("Testing attrition predictor...")
            test_employee = {
                'employee_id': 'TEST001',
                'age': 35,
                'years_at_company': 5,
                'monthly_income': 50000,
                'performance_rating': 3,
                'satisfaction_level': 0.7,
                'last_promotion_years': 2,
                'training_hours': 40
            }
            result = self.attrition_predictor.predict(test_employee)
            logger.info(f"  ✅ Attrition prediction: {result['risk_level']} (probability: {result['attrition_probability']:.2f})")
            
        except Exception as e:
            logger.error(f"  ❌ Attrition test failed: {e}")
            all_passed = False
        
        try:
            # Test forecaster
            logger.info("\nTesting forecaster...")
            forecast = self.forecaster.forecast("Python", months_ahead=3)
            logger.info(f"  ✅ Forecast generated: {len(forecast)} months")
            
        except Exception as e:
            logger.error(f"  ❌ Forecast test failed: {e}")
            all_passed = False
        
        try:
            # Test mobility
            logger.info("\nTesting mobility analyzer...")
            similar = self.mobility_analyzer.find_similar_employees("EMP1000", top_k=3)
            logger.info(f"  ✅ Found {len(similar)} similar employees")
            
        except Exception as e:
            logger.error(f"  ❌ Mobility test failed: {e}")
            all_passed = False
        
        logger.info("\n")
        logger.info("=" * 60)
        if all_passed:
            logger.info("✅ ALL VALIDATIONS PASSED!")
        else:
            logger.error("❌ SOME VALIDATIONS FAILED")
        logger.info("=" * 60)
        logger.info("\n")
        
        return all_passed


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="AWIS Model Loader - Setup and validate all ML models"
    )
    parser.add_argument(
        '--force-retrain',
        action='store_true',
        help='Force retraining of all models even if they exist'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate models after loading'
    )
    parser.add_argument(
        '--data-only',
        action='store_true',
        help='Only generate data, skip model training'
    )
    
    args = parser.parse_args()
    
    loader = ModelLoader()
    
    if args.data_only:
        # Only generate data
        if not loader.check_data_exists():
            loader.generate_data()
        else:
            logger.info("Data already exists. Use --force-retrain to regenerate.")
    else:
        # Full setup
        success = loader.setup_all(force_retrain=args.force_retrain)
        
        # Validate if requested
        if success and args.validate:
            loader.validate_models()


if __name__ == "__main__":
    main()