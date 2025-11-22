from fastapi import APIRouter, HTTPException, Header
from typing import Optional
import pandas as pd
import sys
from pathlib import Path
import time

CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parent.parent.parent
PROJECT_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from ml.inference_attrition import get_predictor
from utils.logger import setup_logger

logger = setup_logger(__name__)

router = APIRouter(prefix="/attrition", tags=["Attrition Prediction"])

# Cache for stats (to avoid recalculating every time)
_stats_cache = None
_cache_time = None

@router.post("/predict")
async def predict_attrition(
    employee_data: dict,
    authorization: Optional[str] = Header(None)
):
    """Predict attrition for a single employee"""
    try:
        predictor = get_predictor()
        prediction = predictor.predict(employee_data)
        recommendations = predictor.get_retention_recommendations(employee_data, prediction)
        prediction['recommendations'] = recommendations
        
        logger.info(f"Attrition prediction for {employee_data.get('employee_id')}: {prediction['risk_level']}")
        
        return prediction
        
    except Exception as e:
        logger.error(f"Error predicting attrition: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/high-risk")
async def get_high_risk_employees(
    threshold: float = 0.7,
    limit: int = 20,
    authorization: Optional[str] = Header(None)
):
    """Get list of high-risk employees"""
    try:
        logger.info(f"Loading high-risk employees (threshold={threshold}, limit={limit})")
        
        employees_path = PROJECT_ROOT / "data" / "employees.csv"
        employees_df = pd.read_csv(employees_path)
        
        # Sample if dataset is large (for performance)
        if len(employees_df) > 500:
            logger.info("Large dataset detected, sampling for performance...")
            employees_df = employees_df.sample(n=500, random_state=42)
        
        predictor = get_predictor()
        
        logger.info(f"Running predictions on {len(employees_df)} employees...")
        start_time = time.time()
        
        predictions_df = predictor.predict_batch(employees_df)
        
        elapsed = time.time() - start_time
        logger.info(f"Predictions complete in {elapsed:.2f}s")
        
        # Filter high risk
        high_risk = predictions_df[
            predictions_df['attrition_probability'] >= threshold
        ].sort_values('attrition_probability', ascending=False)
        
        high_risk = high_risk.head(limit)
        
        # Add recommendations
        results = []
        for _, row in high_risk.iterrows():
            pred_dict = row.to_dict()
            
            emp_data = employees_df[
                employees_df['employee_id'] == row['employee_id']
            ].iloc[0].to_dict()
            
            recommendations = predictor.get_retention_recommendations(emp_data, pred_dict)
            pred_dict['recommendations'] = recommendations
            results.append(pred_dict)
        
        logger.info(f"Retrieved {len(results)} high-risk employees")
        
        return results
        
    except Exception as e:
        logger.error(f"Error getting high-risk employees: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats")
async def get_attrition_stats(authorization: Optional[str] = Header(None)):
    """Get attrition statistics (cached for performance)"""
    global _stats_cache, _cache_time
    
    try:
        # Check cache (refresh every 5 minutes)
        current_time = time.time()
        if _stats_cache and _cache_time and (current_time - _cache_time) < 300:
            logger.info("Returning cached stats")
            return _stats_cache
        
        logger.info("Calculating fresh attrition stats...")
        
        employees_path = PROJECT_ROOT / "data" / "employees.csv"
        employees_df = pd.read_csv(employees_path)
        
        # For large datasets, sample for stats calculation
        sample_size = min(500, len(employees_df))
        if len(employees_df) > sample_size:
            logger.info(f"Sampling {sample_size} employees for stats...")
            sample_df = employees_df.sample(n=sample_size, random_state=42)
        else:
            sample_df = employees_df
        
        predictor = get_predictor()
        
        logger.info(f"Running predictions on {len(sample_df)} employees...")
        start_time = time.time()
        
        predictions_df = predictor.predict_batch(sample_df)
        
        elapsed = time.time() - start_time
        logger.info(f"Predictions complete in {elapsed:.2f}s")
        
        # Calculate stats
        total_employees = len(employees_df)  # Use actual total
        
        # Count by risk level
        high_risk = len(predictions_df[predictions_df['risk_level'] == 'High'])
        medium_risk = len(predictions_df[predictions_df['risk_level'] == 'Medium'])
        low_risk = len(predictions_df[predictions_df['risk_level'] == 'Low'])
        
        # Scale to full population if sampled
        if len(employees_df) > sample_size:
            scale_factor = len(employees_df) / len(sample_df)
            high_risk = int(high_risk * scale_factor)
            medium_risk = int(medium_risk * scale_factor)
            low_risk = int(low_risk * scale_factor)
        
        avg_probability = predictions_df['attrition_probability'].mean()
        
        stats = {
            'total_employees': total_employees,
            'high_risk_count': high_risk,
            'medium_risk_count': medium_risk,
            'low_risk_count': low_risk,
            'high_risk_percentage': (high_risk / total_employees * 100) if total_employees > 0 else 0,
            'average_attrition_probability': float(avg_probability),
            'risk_distribution': {
                'High': high_risk,
                'Medium': medium_risk,
                'Low': low_risk
            },
            'cached': False,
            'sample_size': len(sample_df),
            'calculation_time': f"{elapsed:.2f}s"
        }
        
        # Cache the results
        _stats_cache = stats
        _cache_time = current_time
        
        logger.info(f"Stats calculated successfully in {elapsed:.2f}s")
        
        return stats
        
    except Exception as e:
        logger.error(f"Error calculating attrition stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/clear-cache")
async def clear_stats_cache(authorization: Optional[str] = Header(None)):
    """Clear the stats cache"""
    global _stats_cache, _cache_time
    _stats_cache = None
    _cache_time = None
    return {"message": "Stats cache cleared"}