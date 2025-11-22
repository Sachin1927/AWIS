from langchain.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import sys
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parent.parent.parent
sys.path.insert(0, str(SRC_DIR))

from rag_index.retriever import get_retriever

class RAGInput(BaseModel):
    query: str = Field(description="Question about HR policies")

class HRPolicyTool(BaseTool):
    name = "search_hr_policies"
    description = """
    Search company HR policies and documents.
    Use when users ask about:
    - Company policies
    - Benefits
    - Remote work
    - Promotions
    - Any HR procedures
    
    Input: query
    """
    args_schema: Type[BaseModel] = RAGInput
    
    def _run(self, query: str) -> str:
        try:
            retriever = get_retriever()
            context_data = retriever.get_relevant_context(query, k=3)
            
            result = f"Based on company documents:\n\n{context_data['context']}"
            result += f"\n\nSources: {', '.join(context_data['sources'])}"
            
            return result
        except Exception as e:
            return f"Error searching policies: {str(e)}"
    
    async def _arun(self, query: str) -> str:
        return self._run(query)