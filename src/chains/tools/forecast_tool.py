from langchain.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import sys
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parent.parent.parent
sys.path.insert(0, str(SRC_DIR))

from ml.inference_forecast import get_forecaster

class ForecastInput(BaseModel):
    skill_name: str = Field(description="Skill name like Python, JavaScript, Leadership, Data_Science")

class SkillForecastTool(BaseTool):
    name = "forecast_skill_demand"
    description = """
    Forecast future demand for skills.
    Use when users ask about:
    - Future skill needs
    - Trending skills
    - What skills to learn
    
    Input: skill_name
    """
    args_schema: Type[BaseModel] = ForecastInput
    
    def _run(self, skill_name: str) -> str:
        try:
            forecaster = get_forecaster()
            forecast = forecaster.forecast(skill_name, months_ahead=6)
            
            if len(forecast) == 0:
                return f"No forecast data for {skill_name}"
            
            avg_demand = forecast['forecasted_demand'].mean()
            trend_data = forecaster.get_trending_skills(10)
            trend = next((s for s in trend_data if s['skill_name'] == skill_name), None)
            
            result = f"""
Skill Demand Forecast for {skill_name}:
- Average forecasted demand: {avg_demand:.0f}
- Trend: {trend['trend'] if trend else 'Unknown'}
- Growth rate: {trend['growth_rate']:.1f}% if trend else 'N/A'
"""
            return result
        except Exception as e:
            return f"Error: {str(e)}"
    
    async def _arun(self, skill_name: str) -> str:
        return self._run(skill_name)