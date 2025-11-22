import os
import yaml
from pathlib import Path
from typing import Dict, Any
from pydantic_settings import BaseSettings

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent.parent


class Settings(BaseSettings):
    """Application settings"""
    
    # Paths
    PROJECT_ROOT: Path = PROJECT_ROOT
    CONFIG_PATH: Path = PROJECT_ROOT / "5_configs" / "config.yaml"
    CREDENTIALS_PATH: Path = PROJECT_ROOT / "5_configs" / "credentials.yaml"
    DATA_PATH: Path = PROJECT_ROOT / "data"
    MODELS_PATH: Path = PROJECT_ROOT / "4_models"
    
    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    # Security
    SECRET_KEY: str = "change-this-secret-key-in-production-please"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # RAG
    RAG_EMBEDDINGS_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    RAG_LLM_PROVIDER: str = "ollama"
    RAG_LLM_MODEL: str = "llama2"
    RAG_VECTORSTORE_PATH: Path = PROJECT_ROOT / "2_src" / "rag_index" / "vectorstore"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


def load_config() -> Dict[str, Any]:
    """Load configuration from YAML file"""
    config_path = PROJECT_ROOT / "5_configs" / "config.yaml"
    
    if not config_path.exists():
        # Return default config if file doesn't exist
        return {
            'app': {'name': 'AWIS', 'version': '1.0.0'},
            'api': {'host': '0.0.0.0', 'port': 8000, 'reload': True},
            'security': {
                'secret_key': 'change-this-secret-key',
                'algorithm': 'HS256',
                'access_token_expire_minutes': 30
            },
            'rag': {
                'embeddings': {
                    'model': 'sentence-transformers/all-MiniLM-L6-v2',
                    'device': 'cpu'
                },
                'llm': {
                    'provider': 'ollama',
                    'model': 'llama2',
                    'temperature': 0.7
                }
            }
        }
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def load_credentials() -> Dict[str, Any]:
    """Load credentials from YAML file"""
    cred_path = PROJECT_ROOT / "5_configs" / "credentials.yaml"
    
    if not cred_path.exists():
        # Use template if credentials.yaml doesn't exist
        cred_path = PROJECT_ROOT / "5_configs" / "credentials_template.yaml"
    
    if not cred_path.exists():
        # Return default credentials
        return {
            'admin': {
                'username': 'admin',
                'password': 'admin123',
                'email': 'admin@awis.com'
            },
            'demo_users': [
                {'username': 'hr_manager', 'password': 'hr123', 'role': 'admin'},
                {'username': 'employee', 'password': 'emp123', 'role': 'employee'}
            ]
        }
    
    with open(cred_path, 'r') as f:
        credentials = yaml.safe_load(f)
    
    return credentials


# Global settings instance
settings = Settings()
config = load_config()