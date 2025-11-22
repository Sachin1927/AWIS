import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

# Pytest configuration

def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "requires_ollama: marks tests that require Ollama"
    )

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment"""
    # Ensure data exists
    from src.data_synth.generate_data import SyntheticDataGenerator
    
    data_path = Path(__file__).parent.parent / "data"
    
    if not (data_path / "employees.csv").exists():
        print("\nGenerating test data...")
        generator = SyntheticDataGenerator()
        generator.save_all()