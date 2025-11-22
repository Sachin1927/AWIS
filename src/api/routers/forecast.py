from fastapi import APIRouter, HTTPException, Header
from typing import Optional
import sys
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parent.parent.parent
sys.path.insert(0, str(SRC_DIR))

from ml.inference_forecast import get_forecaster
from utils.logger import setup_logger

logger = setup_logger(__name__)

router = APIRouter(prefix="/forecast", tags=["Skill Demand Forecast"])

@router.post("/skill")
async def forecast_skill_demand(
    request: dict,
    authorization: Optional[str] = Header(None)
):
    """Forecast demand for a specific skill"""
    try:
        skill_name = request.get('skill_name')
        months_ahead = request.get('months_ahead', 6)
        
        forecaster = get_forecaster()
        forecast_df = forecaster.forecast(skill_name, months_ahead)
        
        if len(forecast_df) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No data available for skill: {skill_name}"
            )
        
        trending = forecaster.get_trending_skills(top_n=20)
        trend_info = next(
            (s for s in trending if s['skill_name'] == skill_name),
            None
        )
        
        response = {
            'skill_name': skill_name,
            'months_ahead': months_ahead,
            'forecasts': forecast_df.to_dict('records'),
            'trend': trend_info['trend'] if trend_info else None,
            'growth_rate': trend_info['growth_rate'] if trend_info else None
        }
        
        logger.info(f"Forecast generated for {skill_name}")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error forecasting skill demand: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/trending")
async def get_trending_skills(
    top_n: int = 10,
    authorization: Optional[str] = Header(None)
):
    """Get trending skills"""
    try:
        forecaster = get_forecaster()
        trending = forecaster.get_trending_skills(top_n=top_n)
        return trending
        
    except Exception as e:
        logger.error(f"Error getting trending skills: {e}")
        raise HTTPException(status_code=500, detail=str(e))