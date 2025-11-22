import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import hashlib
import json

def hash_password(password: str) -> str:
    """Hash a password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash"""
    return hash_password(plain_password) == hashed_password

def generate_date_range(start_date: str, end_date: str, freq: str = 'M') -> List[str]:
    """Generate date range"""
    dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    return [d.strftime('%Y-%m-%d') for d in dates]

def calculate_skill_gap(
    current_skills: List[str],
    required_skills: List[str]
) -> Dict[str, Any]:
    """Calculate skill gap between current and required skills"""
    current = set(current_skills)
    required = set(required_skills)
    
    return {
        'missing_skills': list(required - current),
        'matched_skills': list(required & current),
        'extra_skills': list(current - required),
        'match_percentage': len(required & current) / len(required) * 100 if required else 0
    }

def normalize_skill_name(skill: str) -> str:
    """Normalize skill name for comparison"""
    return skill.lower().strip().replace(' ', '_')

def safe_division(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division that handles zero denominator"""
    try:
        return numerator / denominator if denominator != 0 else default
    except:
        return default

def format_number(value: float, decimals: int = 2) -> str:
    """Format number with commas and decimals"""
    return f"{value:,.{decimals}f}"

def calculate_attrition_risk_level(probability: float) -> str:
    """Convert attrition probability to risk level"""
    if probability >= 0.7:
        return "High"
    elif probability >= 0.4:
        return "Medium"
    else:
        return "Low"

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        return super(NumpyEncoder, self).default(obj)