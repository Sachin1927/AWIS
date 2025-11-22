from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional, Dict, Any
from datetime import datetime

# ==================== Auth Schemas ====================

class UserLogin(BaseModel):
    username: str = Field(..., description="Username")
    password: str = Field(..., description="Password")

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class TokenData(BaseModel):
    username: Optional[str] = None
    role: Optional[str] = None

class User(BaseModel):
    username: str
    email: Optional[EmailStr] = None
    role: str = "employee"  # "employee" or "admin"
    disabled: Optional[bool] = False

# ==================== Employee Schemas ====================

class EmployeeBase(BaseModel):
    employee_id: str
    name: str
    email: EmailStr
    department: str
    job_role: str
    age: int
    years_at_company: int
    monthly_income: int
    performance_rating: int
    satisfaction_level: float
    last_promotion_years: int
    training_hours: int

class EmployeeCreate(EmployeeBase):
    pass

class Employee(EmployeeBase):
    attrition: Optional[int] = None
    
    class Config:
        from_attributes = True

# ==================== Attrition Schemas ====================

class AttritionPredictionRequest(BaseModel):
    employee_id: str
    age: int
    years_at_company: int
    monthly_income: int
    performance_rating: int
    satisfaction_level: float
    last_promotion_years: int
    training_hours: int
    department_encoded: Optional[int] = 0

class AttritionPredictionResponse(BaseModel):
    employee_id: str
    attrition_prediction: int
    attrition_probability: float
    risk_level: str
    will_leave: bool
    recommendations: List[str]

# ==================== Forecast Schemas ====================

class ForecastRequest(BaseModel):
    skill_name: str = Field(..., description="Skill to forecast")
    months_ahead: int = Field(default=6, ge=1, le=24, description="Forecast horizon")

class ForecastDataPoint(BaseModel):
    date: str
    skill_name: str
    forecasted_demand: int
    confidence: str

class ForecastResponse(BaseModel):
    skill_name: str
    months_ahead: int
    forecasts: List[ForecastDataPoint]
    trend: Optional[str] = None
    growth_rate: Optional[float] = None

class TrendingSkill(BaseModel):
    skill_name: str
    growth_rate: float
    current_demand: int
    trend: str

# ==================== Mobility Schemas ====================

class SimilarEmployee(BaseModel):
    employee_id: str
    similarity_score: float
    name: Optional[str] = None
    job_role: Optional[str] = None
    department: Optional[str] = None

class CareerPathRecommendation(BaseModel):
    target_role: str
    department: str
    similarity_score: float
    skill_match_percentage: float
    missing_skills: List[str]
    matched_skills: List[str]
    required_skills_count: int

class MobilityRequest(BaseModel):
    employee_id: str
    target_role: Optional[str] = None

class MobilityResponse(BaseModel):
    employee_id: str
    current_skills: List[str]
    similar_employees: List[SimilarEmployee]
    career_recommendations: List[CareerPathRecommendation]

# ==================== RAG Schemas ====================

class RAGQuery(BaseModel):
    query: str = Field(..., description="Question about HR policies")
    k: int = Field(default=3, ge=1, le=10, description="Number of documents to retrieve")

class RAGResponse(BaseModel):
    query: str
    answer: str
    sources: List[str]
    context: str
    num_docs: int

# ==================== Agent Schemas ====================

class ChatMessage(BaseModel):
    message: str = Field(..., description="User message")
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    success: bool
    intermediate_steps: Optional[List[Any]] = []
    timestamp: str

# ==================== Analytics Schemas ====================

class DashboardStats(BaseModel):
    total_employees: int
    attrition_rate: float
    avg_satisfaction: float
    high_risk_employees: int
    trending_skills: List[TrendingSkill]

class SkillGapAnalysis(BaseModel):
    employee_id: str
    current_skills: List[str]
    required_skills: List[str]
    missing_skills: List[str]
    match_percentage: float