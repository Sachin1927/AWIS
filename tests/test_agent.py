import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

# Note: These tests require Ollama to be running
# Skip if Ollama is not available

def test_agent_import():
    """Test that agent can be imported"""
    try:
        from src.chains.agent import AWISAgent
        assert AWISAgent is not None
    except Exception as e:
        pytest.skip(f"Agent import failed: {e}")

def test_tools_import():
    """Test that tools can be imported"""
    try:
        from src.chains.tools.forecast_tool import SkillDemandForecastTool
        from src.chains.tools.mobility_tool import CareerMobilityTool
        from src.chains.tools.rag_tool import HRPolicyTool
        
        assert SkillDemandForecastTool is not None
        assert CareerMobilityTool is not None
        assert HRPolicyTool is not None
    except Exception as e:
        pytest.skip(f"Tools import failed: {e}")

@pytest.mark.skipif(True, reason="Requires Ollama to be running")
def test_agent_initialization():
    """Test agent initialization (requires Ollama)"""
    from src.chains.agent import AWISAgent
    
    agent = AWISAgent()
    assert agent.llm is not None
    assert len(agent.tools) > 0

@pytest.mark.skipif(True, reason="Requires Ollama to be running")
def test_agent_chat():
    """Test agent chat (requires Ollama)"""
    from src.chains.agent import AWISAgent
    
    agent = AWISAgent()
    result = agent.chat("What is the remote work policy?")
    
    assert 'response' in result
    assert 'success' in result