from langchain.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import sys
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parent.parent.parent
sys.path.insert(0, str(SRC_DIR))

from ml.inference_attrition import get_predictor

class AttritionInput(BaseModel):
    employee_id: str = Field(description="Employee ID like EMP1001")

class AttritionPredictionTool(BaseTool):
    name = "predict_attrition"
    description = """
    Predict employee attrition risk.
    Use when users ask about:
    - Employee turnover risk
    - Who might leave
    - Retention concerns
    
    Input: employee_id
    """
    args_schema: Type[BaseModel] = AttritionInput
    
    def _run(self, employee_id: str) -> str:
        try:
            predictor = get_predictor()
            
            # Load employee data
            import pandas as pd
            from pathlib import Path
            PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
            emp_df = pd.read_csv(PROJECT_ROOT / "data" / "employees.csv")
            
            emp_data = emp_df[emp_df['employee_id'] == employee_id]
            if len(emp_data) == 0:
                return f"Employee {employee_id} not found"
            
            emp_dict = emp_data.iloc[0].to_dict()
            result = predictor.predict(emp_dict)
            
            return f"""
Employee {employee_id} Attrition Analysis:
- Risk Level: {result['risk_level']}
- Probability: {result['attrition_probability']:.1%}
- Prediction: {'Will likely leave' if result['will_leave'] else 'Will likely stay'}
"""
        except Exception as e:
            return f"Error: {str(e)}"
    
    async def _arun(self, employee_id: str) -> str:
        return self._run(employee_id)