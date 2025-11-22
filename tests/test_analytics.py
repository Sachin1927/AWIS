import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.api.main import app

client = TestClient(app)

@pytest.fixture
def auth_headers():
    """Get authentication headers"""
    response = client.post(
        "/auth/login",
        data={
            "username": "admin",
            "password": "admin123"
        }
    )
    
    assert response.status_code == 200
    token = response.json()["access_token"]
    
    return {"Authorization": f"Bearer {token}"}

def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_login():
    """Test login endpoint"""
    response = client.post(
        "/auth/login",
        data={
            "username": "admin",
            "password": "admin123"
        }
    )
    
    assert response.status_code == 200
    assert "access_token" in response.json()
    assert response.json()["token_type"] == "bearer"

def test_login_invalid_credentials():
    """Test login with invalid credentials"""
    response = client.post(
        "/auth/login",
        data={
            "username": "invalid",
            "password": "wrong"
        }
    )
    
    assert response.status_code == 401

def test_get_current_user(auth_headers):
    """Test getting current user info"""
    response = client.get("/auth/me", headers=auth_headers)
    
    assert response.status_code == 200
    data = response.json()
    assert "username" in data
    assert "role" in data

def test_attrition_stats(auth_headers):
    """Test attrition statistics endpoint"""
    response = client.get("/attrition/stats", headers=auth_headers)
    
    assert response.status_code == 200
    data = response.json()
    assert "total_employees" in data
    assert "high_risk_count" in data

def test_attrition_prediction(auth_headers):
    """Test attrition prediction endpoint"""
    employee_data = {
        "employee_id": "TEST001",
        "age": 35,
        "years_at_company": 5,
        "monthly_income": 50000,
        "performance_rating": 3,
        "satisfaction_level": 0.7,
        "last_promotion_years": 2,
        "training_hours": 40
    }
    
    response = client.post(
        "/attrition/predict",
        json=employee_data,
        headers=auth_headers
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "attrition_probability" in data
    assert "risk_level" in data

def test_trending_skills(auth_headers):
    """Test trending skills endpoint"""
    response = client.get(
        "/forecast/trending?top_n=5",
        headers=auth_headers
    )
    
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)

def test_skill_forecast(auth_headers):
    """Test skill forecast endpoint"""
    forecast_data = {
        "skill_name": "Python",
        "months_ahead": 6
    }
    
    response = client.post(
        "/forecast/skill",
        json=forecast_data,
        headers=auth_headers
    )
    
    # May return 404 if skill not found, which is acceptable
    assert response.status_code in [200, 404]

def test_unauthorized_access():
    """Test accessing protected endpoint without auth"""
    response = client.get("/attrition/stats")
    assert response.status_code == 401