import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.rag_index.retriever import RAGRetriever

@pytest.fixture
def retriever():
    """Create retriever instance"""
    try:
        return RAGRetriever()
    except FileNotFoundError:
        pytest.skip("Vector store not built yet")

def test_retriever_loads(retriever):
    """Test that retriever loads successfully"""
    assert retriever.vectorstore is not None
    assert retriever.embeddings is not None

def test_retrieve_documents(retriever):
    """Test document retrieval"""
    docs = retriever.retrieve("remote work policy", k=3)
    
    assert isinstance(docs, list)
    assert len(docs) <= 3
    
    if len(docs) > 0:
        assert hasattr(docs[0], 'page_content')
        assert hasattr(docs[0], 'metadata')

def test_retrieve_with_scores(retriever):
    """Test retrieval with similarity scores"""
    docs_and_scores = retriever.retrieve_with_scores("promotion criteria", k=2)
    
    assert isinstance(docs_and_scores, list)
    
    if len(docs_and_scores) > 0:
        doc, score = docs_and_scores[0]
        assert hasattr(doc, 'page_content')
        assert isinstance(score, (int, float))

def test_get_relevant_context(retriever):
    """Test context extraction"""
    context_data = retriever.get_relevant_context("What is the training budget?", k=2)
    
    assert 'context' in context_data
    assert 'sources' in context_data
    assert 'num_docs' in context_data
    assert isinstance(context_data['context'], str)
    assert isinstance(context_data['sources'], list)

def test_format_docs(retriever):
    """Test document formatting"""
    docs = retriever.retrieve("performance review", k=2)
    
    if len(docs) > 0:
        formatted = retriever.format_docs_for_context(docs)
        assert isinstance(formatted, str)
        assert len(formatted) > 0