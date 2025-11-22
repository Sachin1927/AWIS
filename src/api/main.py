from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys
from pathlib import Path
import uvicorn

CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parent.parent
PROJECT_ROOT = SRC_DIR.parent

sys.path.insert(0, str(SRC_DIR))

from api.routers import auth, attrition, forecast, mobility, chat
from utils.logger import setup_logger

logger = setup_logger(__name__)

app = FastAPI(
    title="AWIS API",
    description="Adaptive Workforce Intelligence System",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router)
app.include_router(attrition.router)
app.include_router(forecast.router)
app.include_router(mobility.router)
app.include_router(chat.router)

@app.get("/")
async def root():
    return {
        "message": "Welcome to AWIS API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "AWIS API"}

if __name__ == "__main__":
    logger.info("="*60)
    logger.info("🚀 Starting AWIS API...")
    logger.info("="*60)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )