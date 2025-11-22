from langchain.tools import BaseTool
from typing import Type, Optional
from pydantic import BaseModel, Field
import sys
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parent.parent.parent
sys.path.insert(0, str(SRC_DIR))

from ml.inference_mobility import get_analyzer

class MobilityInput(BaseModel):
    """Input schema for mobility tool"""
    employee_id: str = Field(description="Employee ID like EMP1001")

class CareerMobilityTool(BaseTool):
    """Tool for career mobility recommendations"""
    
    name: str = "career_recommendations"
    description: str = """
    Get career path recommendations for employees.
    Use when users ask about:
    - Career growth opportunities
    - Skill gaps for advancement
    - Promotion paths
    - Career development
    
    Input should be an employee ID (e.g., EMP1001)
    """
    args_schema: Type[BaseModel] = MobilityInput
    
    def _run(self, employee_id: str) -> str:
        """Execute the tool"""
        try:
            analyzer = get_analyzer()
            
            # Get career recommendations
            recommendations = analyzer.recommend_career_paths(employee_id, target_role=None)
            
            if not recommendations or len(recommendations) == 0:
                return f"No career recommendations found for employee {employee_id}. Employee may not exist in the system."
            
            # Format results
            result = f"Career Path Recommendations for {employee_id}:\n\n"
            
            for i, rec in enumerate(recommendations[:3], 1):
                result += f"{i}. {rec['target_role']} ({rec['department']})\n"
                result += f"   • Skill match: {rec['skill_match_percentage']:.1f}%\n"
                result += f"   • Missing skills: {len(rec['missing_skills'])} skills needed\n"
                
                if rec['missing_skills']:
                    top_missing = rec['missing_skills'][:3]
                    result += f"   • Top gaps: {', '.join(top_missing)}\n"
                
                result += "\n"
            
            return result
            
        except FileNotFoundError as e:
            return f"Error: Model files not found. Please ensure mobility model is trained. Details: {str(e)}"
        except Exception as e:
            return f"Error analyzing career mobility: {str(e)}"
    
    async def _arun(self, employee_id: str) -> str:
        """Async version"""
        return self._run(employee_id)