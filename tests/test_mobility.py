import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.ml.inference_mobility import MobilityAnalyzer

@pytest.fixture
def analyzer():
    """Create analyzer instance"""
    return MobilityAnalyzer()

def test_analyzer_loads(analyzer):
    """Test that analyzer loads successfully"""
    assert analyzer.graph is not None
    assert analyzer.model is not None
    assert analyzer.embeddings is not None

def test_find_similar_employees(analyzer):
    """Test finding similar employees"""
    # Use employee that exists
    similar = analyzer.find_similar_employees("EMP1000", top_k=3)
    
    assert isinstance(similar, list)
    
    if len(similar) > 0:
        assert len(similar) <= 3
        
        for emp in similar:
            assert 'employee_id' in emp
            assert 'similarity_score' in emp
            assert 0 <= emp['similarity_score'] <= 1

def test_get_employee_skills(analyzer):
    """Test getting employee skills"""
    skills = analyzer.get_employee_skills("EMP1000")
    
    assert isinstance(skills, list)

def test_career_path_recommendations(analyzer):
    """Test career path recommendations"""
    recommendations = analyzer.recommend_career_paths("EMP1000")
    
    assert isinstance(recommendations, list)
    
    if len(recommendations) > 0:
        for rec in recommendations:
            assert 'target_role' in rec
            assert 'skill_match_percentage' in rec
            assert 'missing_skills' in rec
            assert isinstance(rec['missing_skills'], list)

def test_skills_for_role(analyzer):
    """Test skill recommendations for role"""
    skills = analyzer.recommend_skills_for_role(
        "Software Engineer",
        top_k=5
    )
    
    assert isinstance(skills, list)
    
    if len(skills) > 0:
        assert len(skills) <= 5
        
        for skill in skills:
            assert 'skill_name' in skill
            assert 'frequency' in skill
            assert 'importance' in skill

def test_graph_metrics(analyzer):
    """Test graph metrics calculation"""
    metrics = analyzer.get_skill_graph_metrics()
    
    assert 'total_employees' in metrics
    assert 'total_skills' in metrics
    assert 'total_connections' in metrics
    assert metrics['total_employees'] > 0
    assert metrics['total_skills'] > 0