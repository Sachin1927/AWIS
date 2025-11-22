from fastapi import APIRouter, Depends, HTTPException
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.api.models.pydantic_schemas import (
    RAGQuery,
    RAGResponse,
    ChatMessage,
    ChatResponse,
    User
)
from src.api.routers.auth import get_current_active_user
from src.rag_index.retriever import get_retriever
from src.chains.agent import get_agent
from src.utils.logger import setup_logger
from datetime import datetime

logger = setup_logger(__name__)

router = APIRouter(prefix="/rag", tags=["RAG & Chat"])

@router.post("/query", response_model=RAGResponse)
async def query_documents(
    request: RAGQuery,
    current_user: User = Depends(get_current_active_user)
):
    """Query HR documents using RAG"""
    try:
        retriever = get_retriever()
        
        # Get relevant context
        context_data = retriever.get_relevant_context(
            request.query,
            k=request.k
        )
        
        response = {
            'query': request.query,
            'answer': f"Based on the retrieved documents:\n\n{context_data['context']}",
            'sources': context_data['sources'],
            'context': context_data['context'],
            'num_docs': context_data['num_docs']
        }
        
        logger.info(f"RAG query processed: {request.query[:50]}...")
        
        return RAGResponse(**response)
        
    except Exception as e:
        logger.error(f"Error in RAG query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat", response_model=ChatResponse)
async def chat_with_agent(
    request: ChatMessage,
    current_user: User = Depends(get_current_active_user)
):
    """Chat with AWIS agent"""
    try:
        agent = get_agent()
        
        # Process message
        result = agent.chat(request.message)
        
        response = {
            'response': result['response'],
            'success': result['success'],
            'intermediate_steps': result.get('intermediate_steps', []),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Agent chat processed for user {current_user.username}")
        
        return ChatResponse(**response)
        
    except Exception as e:
        logger.error(f"Error in agent chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/reset-chat")
async def reset_chat_session(
    current_user: User = Depends(get_current_active_user)
):
    """Reset chat session (clear memory)"""
    try:
        agent = get_agent()
        agent.reset_memory()
        
        return {"message": "Chat session reset successfully"}
        
    except Exception as e:
        logger.error(f"Error resetting chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))