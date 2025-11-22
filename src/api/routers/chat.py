from fastapi import APIRouter, HTTPException, Header, BackgroundTasks
from typing import Optional
import sys
from pathlib import Path
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parent.parent.parent
sys.path.insert(0, str(SRC_DIR))

from utils.logger import setup_logger

logger = setup_logger(__name__)

router = APIRouter(prefix="/chat", tags=["AI Chat"])

# Thread pool for running blocking agent calls
executor = ThreadPoolExecutor(max_workers=4)

def run_agent_chat(message: str) -> dict:
    """Run agent chat in thread pool"""
    try:
        from chains.agent import get_agent
        agent = get_agent()
        return agent.chat(message)
    except Exception as e:
        logger.error(f"Agent error: {e}")
        return {
            'response': f"Error: {str(e)}",
            'success': False
        }

@router.post("/message")
async def chat_message(request: dict, authorization: Optional[str] = Header(None)):
    """Send message to AI agent with timeout"""
    try:
        message = request.get('message', '').strip()
        
        if not message:
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        logger.info(f"Chat request: {message[:50]}...")
        
        # Run with timeout (30 seconds max)
        try:
            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(executor, run_agent_chat, message),
                timeout=30.0  # 30 second timeout
            )
            
            return {
                'response': result['response'],
                'success': result['success'],
                'timestamp': datetime.now().isoformat()
            }
            
        except asyncio.TimeoutError:
            logger.error("Agent timeout after 30s")
            return {
                'response': "⏱️ Response took too long. Please try a simpler question or try again.",
                'success': False,
                'timestamp': datetime.now().isoformat()
            }
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        return {
            'response': f"System error: {str(e)}",
            'success': False,
            'timestamp': datetime.now().isoformat()
        }

@router.post("/reset")
async def reset_conversation(authorization: Optional[str] = Header(None)):
    """Reset conversation memory"""
    try:
        from chains.agent import get_agent
        agent = get_agent()
        agent.reset_memory()
        return {"message": "Conversation reset successfully"}
    except Exception as e:
        logger.error(f"Reset error: {e}")
        raise HTTPException(status_code=500, detail=str(e))