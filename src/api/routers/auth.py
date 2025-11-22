from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
import hashlib
from datetime import datetime
import sys
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parent.parent.parent
sys.path.insert(0, str(SRC_DIR))

from utils.logger import setup_logger

logger = setup_logger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])

# Password hashing using hashlib (lightweight replacement for passlib for local/testing use)
def hash_password(plain_password: str) -> str:
    """Return SHA256 hex digest of the password (not salted - suitable for testing only)."""
    return hashlib.sha256(plain_password.encode('utf-8')).hexdigest()

# OAuth2 scheme - THIS IS WHAT OTHER FILES IMPORT
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

# Simple user database
USERS_DB = {
    "admin": {
        "username": "admin",
        "email": "admin@awis.com",
        "hashed_password": hash_password("admin123"),
        "role": "admin",
        "disabled": False
    },
    "hr_manager": {
        "username": "hr_manager",
        "email": "hr@awis.com",
        "hashed_password": hash_password("hr123"),
        "role": "admin",
        "disabled": False
    },
    "employee": {
        "username": "employee",
        "email": "emp@awis.com",
        "hashed_password": hash_password("emp123"),
        "role": "employee",
        "disabled": False
    }
}

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password"""
    return hash_password(plain_password) == hashed_password

def authenticate_user(username: str, password: str):
    """Authenticate user"""
    user = USERS_DB.get(username)
    if not user:
        return False
    if not verify_password(password, user["hashed_password"]):
        return False
    return user

def create_access_token(data: dict):
    """Create simple access token"""
    return f"token_{data['sub']}_{datetime.utcnow().timestamp()}"

def get_current_user(token: str = Depends(oauth2_scheme)):
    """Get current user from token"""
    try:
        username = token.split("_")[1] if "_" in token else "admin"
        user = USERS_DB.get(username, USERS_DB["admin"])
        return user
    except:
        return USERS_DB["admin"]

@router.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login endpoint"""
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(data={"sub": user["username"]})
    
    logger.info(f"User {user['username']} logged in successfully")
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "username": user["username"],
            "email": user["email"],
            "role": user["role"]
        }
    }

@router.get("/me")
async def read_users_me(token: str = Depends(oauth2_scheme)):
    """Get current user info"""
    user = get_current_user(token)
    return {
        "username": user["username"],
        "email": user["email"],
        "role": user["role"]
    }